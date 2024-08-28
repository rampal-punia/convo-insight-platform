import json

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from .utils.text_chat_handler import TextChatHandler
from .models import Conversation, Message, UserText, AIText
# from .tasks import process_ai_response, process_user_message, analyze_conversation_sentiment


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        '''Accept the connections from front-end'''
        # get the user from scope
        self.user = self.scope['user']
        # check if the user is authenticated
        if not self.user.is_authenticated:
            await self.close()
            return

        # Get the conversation UUID from the url route
        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')

        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! you are now connected to the convo-chat."
        }))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None):
        '''Run on receiving text data from front-end.'''
        input_data, conversation, user_message = await self.create_conversation_db(text_data)

        await TextChatHandler.process_text_response(
            conversation,
            user_message,
            input_data,
            self.send
        )

    async def create_conversation_db(self, text_data):
        data = json.loads(text_data)
        input_type = data.get('type')   # front-end message
        input_data = data.get('message')
        uuid = data.get('uuid')

        conversation = await self.get_or_create_conversation(uuid)
        user_message = await self.save_message(conversation, input_type, is_from_user=True)
        await self.save_usertext(user_message, input_data)
        return input_data, conversation, user_message

    @database_sync_to_async
    def get_or_create_conversation(self, uuid):
        conversation, created = Conversation.objects.update_or_create(
            id=uuid,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )
        if created:
            Conversation.objects.filter(user=self.user, status='AC').exclude(
                id=uuid
            ).update(status='EN')
        return conversation

    @database_sync_to_async
    def save_usertext(self, message, input_data):
        return UserText.objects.create(
            message=message,
            content=input_data,
        )

    # @database_sync_to_async
    # def process_user_message(self, content):
    #     # Perform immediate analysis
    #     sentiment_score = analyze_sentiment(content)
    #     intent = predict_intent(content)
    #     topics = generate_topic_distribution(content)

    #     # Save UserText with analysis results
    #     user_text = UserText.objects.create(
    #         conversation=self.conversation,
    #         content=content,
    #         is_from_user=True,
    #         sentiment_score=sentiment_score,
    #         intent=intent,
    #         # Set the highest weighted topic as primary
    #         primary_topic=max(topics, key=lambda x: x[1])[0]
    #     )

    #     # Save topic distributions
    #     for topic, weight in topics:
    #         TopicDistribution.objects.create(
    #             user_text=user_text,
    #             topic=topic,
    #             weight=weight
    #         )

    #     return user_text

    @database_sync_to_async
    def save_aitext(self, message, input_data):
        return AIText.objects.create(
            message=message,
            content=input_data,
        )

    @database_sync_to_async
    def save_message(self, conversation, content_type, is_from_user=True, in_reply_to=None):
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )
