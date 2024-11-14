import json
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from langchain_core.messages import HumanMessage, AIMessage
from channels.db import database_sync_to_async
from django.core.files.base import ContentFile

from .models import GeneralConversation, GeneralMessage, AudioMessage
from convochat.utils import configure_llm
from .models import AudioMessage
from .services import VoiceModalHandler


class GeneralChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope['user']
        if not self.user.is_authenticated:
            await self.close()
            return

        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')
        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! You are now connected to the General-Assistant."
        }))

        self.llm = configure_llm.get_chat_llm()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        input_type = data.get('type')
        input_data = data.get('message')
        conversation_id = data.get('uuid')

        # save the conversation database instance
        self.conversation_dbi = await self.get_or_create_conversation(conversation_id)

        if input_type == 'TE':
            user_message_dbi = await self.save_message(input_type, input_data, is_from_user=True)
            await self.process_text_response(
                user_message_dbi,
                input_data,
            )
        elif input_type == 'AU':
            await self.handle_audio_message(input_data)

    async def process_text_response(
        self,
        user_message_dbi,
        input_data,
    ):
        try:
            history = await self.get_conversation_history()
            print("history: ", history)
            history_str = '\n'.join(
                [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}:{msg.content}" for msg in history]
            )
            input_with_history = {
                'history': history_str,
                'input': input_data
            }
            print("input_with_history: ", input_with_history)

            # Generate AI response
            llm_response_chunks = []
            async for chunk in self.llm.astream_events(input_with_history, version='v2'):
                if chunk['event'] in ['on_parser_start', 'on_parser_stream']:
                    await self.send(text_data=json.dumps(chunk))

                if chunk.get('event') == 'on_parser_end':
                    output = chunk.get('data', {}).get('output', '')
                    llm_response_chunks.append(output)

            ai_response = ''.join(llm_response_chunks)

            # Generate and update title
            if self.conversation_dbi.title == 'Untitled Conversation' or self.conversation_dbi.title is None:
                try:
                    new_title = await configure_llm.generate_title(ai_response)
                    await self.save_conversation_title(new_title)
                    await self.send(text_data=json.dumps({
                        'type': 'title_update',
                        'title': new_title
                    }))
                except Exception as ex:
                    print(f"Unable to generate title: {ex}")

            if not ai_response:
                ai_response = "I apologize, but I couldn't generate a response. Please try asking your question again."

            await self.save_message(
                content_type='TE',
                content=ai_response,
                is_from_user=False,
                in_reply_to=user_message_dbi
            )
            return ai_response
        except Exception as ex:
            print(f"Unable to process response: {ex}")

    async def handle_audio_message(self, audio_data):
        voice_handler = VoiceModalHandler()

        # Decode base64 audio data
        audio_content = base64.b64decode(audio_data)

        # Save user's audio message
        user_message_dbi = await self.save_message(content_type='AU', is_from_user=True)
        await self.save_audio_message(user_message_dbi, audio_content)

        # Process user audio (speech-to-text)
        text_content = await voice_handler.process_audio(user_message_dbi)

        # Send transcription to the client
        await self.send(text_data=json.dumps({
            'type': 'transcription',
            'message': text_content,
            'id': str(user_message_dbi.id)
        }))

        # Generate AI response
        ai_response = await self.process_text_response(
            user_message_dbi,
            text_content,
        )

        # Convert AI response to speech
        ai_audio = await voice_handler.text_to_speech(ai_response)

        # Save AI's audio message
        ai_message_dbi = await self.save_message(content_type='AU', is_from_user=False, in_reply_to=user_message_dbi)
        await self.save_audio_message(ai_message_dbi, ai_audio, ai_response)

        # Send AI response to the client
        await self.send(text_data=json.dumps({
            'type': 'ai_response',
            'message': ai_response,
            'audio_url': ai_message_dbi.audio_content.audio_file.url,
            'id': str(ai_message_dbi.id)
        }))

    @database_sync_to_async
    def get_or_create_conversation(self, conversation_id):
        conversation_dbi, created = GeneralConversation.objects.update_or_create(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )
        if created:
            GeneralConversation.objects.filter(user=self.user, status='AC').exclude(
                id=conversation_id).update(status='EN')
        return conversation_dbi

    @database_sync_to_async
    def save_message(self, content_type, content='No Text Data Found', is_from_user=True, in_reply_to=None):
        return GeneralMessage.objects.create(
            conversation=self.conversation_dbi,
            content_type=content_type,
            content=content,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )

    @database_sync_to_async
    def save_audio_message(self, message, audio_content, transcript=''):
        print('transcript: ', transcript)
        audio_file = ContentFile(audio_content, name=f"audio_{message.id}.wav")
        return AudioMessage.objects.create(
            message=message,
            audio_file=audio_file,
            transcript=transcript,  # This will be updated after processing
            duration=0.0  # This will be updated after processing
        )

    @database_sync_to_async
    def save_conversation_title(self, title):
        self.conversation_dbi.title = title
        self.conversation_dbi.save()

    @database_sync_to_async
    def get_conversation_history(self, limit=4):
        messages = self.conversation_dbi.general_messages.order_by(
            '-created')[:limit]
        return [
            HumanMessage(content=msg.content) if msg.is_from_user else AIMessage(
                content=msg.content)
            for msg in reversed(messages)
        ]
