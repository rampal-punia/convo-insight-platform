import aiohttp
import json
from django.conf import settings
from langchain_core.messages import HumanMessage, AIMessage
from channels.db import database_sync_to_async
from convochat.models import Conversation, Message, UserText
from apps.convochat.utils import configure_llm


# Summary generation API
# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

# Title generation API
API_URL = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_TOKEN}"}


@database_sync_to_async
def save_conversation_title(conversation, title):
    conversation.title = title
    conversation.save()


@database_sync_to_async
def get_conversation_history(conversation_id, limit=8):
    conversation = Conversation.objects.get(id=conversation_id)
    messages = conversation.messages.order_by('-created')[:limit]
    return [
        HumanMessage(content=msg.chat_content.content) if msg.is_from_user else AIMessage(
            content=msg.chat_content.content)
        for msg in reversed(messages)
    ]


@database_sync_to_async
def save_message(conversation, content_type, is_from_user=True, in_reply_to=None):
    return Message.objects.create(
        conversation=conversation,
        content_type=content_type,
        is_from_user=is_from_user,
        in_reply_to=in_reply_to
    )


@database_sync_to_async
def save_chat_message(message, input_data):
    return UserText.objects.create(
        message=message,
        content=input_data,
    )


async def generate_title(conversation_content):
    async with aiohttp.ClientSession() as session:
        async with session.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": conversation_content,
                    "parameters": {"max_length": 50, "min_length": 10}
                }) as response:
            result = await response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]['generated_text']
            else:
                return "Untitled Conversation"


class TextChatHandler:
    @staticmethod
    async def process_text_response(
        conversation,   # User-Ai conversation db table instance
        user_message,   # user_message db table instance
        input_data,     # Front end user message
        send_method,     # self.send method of consumer
        # In case of RAG query related cosine similary bases retrived context from vectordb
        context=None,
        summary=None
    ):
        try:
            if context:
                input_with_history_and_context = {
                    'history': summary,
                    'input': input_data,
                    'context': context
                }

                # Generate AI response
                llm_response_chunks = []
                async for chunk in configure_llm.doc_chain.astream_events(input_with_history_and_context, version='v2', include_names=['Assistant']):
                    if chunk['event'] in ['on_parser_start', 'on_parser_stream']:
                        await send_method(text_data=json.dumps(chunk))

                    if chunk.get('event') == 'on_parser_end':
                        output = chunk.get('data', {}).get('output', '')
                        llm_response_chunks.append(output)
            else:
                history = await get_conversation_history(conversation.id)
                history_str = '\n'.join(
                    [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}:{msg.content}" for msg in history]
                )
                input_with_history = {
                    'history': history_str,
                    'input': input_data
                }
                # Generate AI response
                llm_response_chunks = []
                async for chunk in configure_llm.chain.astream_events(input_with_history, version='v2', include_names=['Assistant']):
                    if chunk['event'] in ['on_parser_start', 'on_parser_stream']:
                        await send_method(text_data=json.dumps(chunk))

                    if chunk.get('event') == 'on_parser_end':
                        output = chunk.get('data', {}).get('output', '')
                        llm_response_chunks.append(output)

            ai_response = ''.join(llm_response_chunks)

            # Generate and update title
            if conversation.title == 'Untitled Conversation' or conversation.title is None:
                try:
                    new_title = await generate_title(ai_response)
                    await save_conversation_title(conversation, new_title)
                    await send_method(text_data=json.dumps({
                        'type': 'title_update',
                        'title': new_title
                    }))
                except Exception as ex:
                    print(f"Unable to generate title: {ex}")
            if not ai_response:
                ai_response = "I apologize, but I couldn't generate a response. Please try asking your question again."

            ai_message = await save_message(
                conversation,
                content_type='TE',
                is_from_user=False,
                in_reply_to=user_message
            )
            await save_chat_message(
                ai_message,
                ai_response
            )
        except Exception as ex:
            print(f"Unable to process response: {ex}")
