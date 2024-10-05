import json
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from decimal import Decimal
from convochat.models import Conversation, Message, UserText, AIText
from django.forms.models import model_to_dict
from .models import Order, OrderConversationLink, OrderItem
# from .tasks import process_ai_response, process_user_message, analyze_conversation_sentiment
from .tasks import analyze_sentiment, recognize_intent
import aiohttp
import json
from django.conf import settings
from langchain_core.messages import HumanMessage, AIMessage
from channels.db import database_sync_to_async
from convochat.models import Conversation, Message, AIText
from apps.convochat.utils import configure_llm

# print("*"*40)
# print("order id is: ", order_id)

# Title generation API
API_URL = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}


@database_sync_to_async
def save_conversation_title(conversation, title):
    conversation.title = title
    conversation.save()


@database_sync_to_async
def get_conversation_history(conversation_id, limit=8):
    conversation = Conversation.objects.get(id=conversation_id)
    messages = conversation.messages.order_by('-created')[:limit]
    return [
        HumanMessage(content=msg.user_text.content) if msg.is_from_user else AIMessage(
            content=msg.ai_text.content)
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
def save_aitext(message, input_data):
    return AIText.objects.create(
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


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, '__str__'):  # For model instances or other objects
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class OrderSupportConsumer(AsyncWebsocketConsumer):
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
            'message': f"Welcome, {self.user}! you are now connected to the Order Support"
        }))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None):
        '''Run on receiving text data from front-end.'''
        input_data, conversation, user_message, order_id, order = await self.create_conversation_db(text_data)

        # Process the message in real-time
        # analyze_topics.delay(conversation.id)
        predicted_intent = recognize_intent.delay(user_message.id)
        predicted_sentiment = analyze_sentiment.delay(user_message.id)

        # Get order details asynchronously
        order_details = await self.get_order_details(order_id)

        await self.process_text_response(
            conversation,
            user_message,
            input_data,
            order_details,
        )

    async def get_order_details(self, order_id):
        # Fetch the order asynchronously
        order = await sync_to_async(Order.objects.get)(id=order_id)

        # Convert the order to a dict asynchronously
        order_dict = await sync_to_async(model_to_dict)(order)

        # Fetch related items asynchronously
        items = await sync_to_async(list)(order.items.all())

        # Convert items to dicts
        item_dicts = await sync_to_async(lambda: [model_to_dict(item) for item in items])()

        # Add items to the order dict
        order_dict['items'] = item_dicts

        # Add full status description
        order_dict['status_description'] = await sync_to_async(order.get_status_display)()

        # Add all possible statuses for reference
        order_dict['all_statuses'] = dict(Order.Status.choices)
        print("*"*40)
        print("order dict is: ", order_dict)

        # Convert to JSON-compatible format
        json_compatible_dict = json.loads(
            json.dumps(order_dict, default=decimal_default)
        )
        print("*"*40)
        print("order json_compatible_dict is: ", json_compatible_dict)

        return json_compatible_dict

    async def process_text_response(
        self,
        conversation,   # User-Ai conversation db table instance
        user_message,   # user_message db table instance
        input_data,     # Front end user message
        order_dict=None,
    ):
        try:
            history = await get_conversation_history(conversation.id)
            history_str = '\n'.join(
                [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}:{msg.content}" for msg in history]
            )
            input_with_order_history = {
                'history': history_str,
                'input': input_data,
                # Convert to formatted JSON string
                'order_dict': json.dumps(order_dict, indent=2)
            }

            # Generate AI response
            llm_response_chunks = []
            async for chunk in configure_llm.order_main().astream_events(input_with_order_history, version='v2', include_names=['Assistant']):
                if chunk['event'] in ['on_parser_start', 'on_parser_stream']:
                    await self.send(text_data=json.dumps(chunk))

                if chunk.get('event') == 'on_parser_end':
                    output = chunk.get('data', {}).get('output', '')
                    llm_response_chunks.append(output)
            ai_response = ''.join(llm_response_chunks)
            print(ai_response)

            if not ai_response:
                ai_response = "I apologize, but I couldn't generate a response. Please try asking your question again."

            ai_message = await save_message(
                conversation,
                content_type='TE',
                is_from_user=False,
                in_reply_to=user_message
            )
            await save_aitext(
                ai_message,
                ai_response
            )
        except Exception as ex:
            print(f"Unable to process response: {ex}")

    async def create_conversation_db(self, text_data):
        data = json.loads(text_data)
        input_type = data.get('type')   # front-end message
        input_data = data.get('message')
        conversation_id = data.get('uuid')
        order_id = data.get('order_id')

        conversation, order = await self.get_or_create_conversation(conversation_id, order_id)
        user_message = await self.save_message(conversation, input_type, is_from_user=True)
        await self.save_usertext(user_message, input_data)
        return input_data, conversation, user_message, order_id, order

    @database_sync_to_async
    def get_or_create_conversation(self, conversation_id, order_id):
        conversation, created = Conversation.objects.update_or_create(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )
        order = Order.objects.get(id=order_id)
        OrderConversationLink.objects.get_or_create(
            order=order,
            conversation=conversation
        )
        if created:
            Conversation.objects.filter(user=self.user, status='AC').exclude(
                id=conversation_id
            ).update(status='EN')
        return conversation, order

    @database_sync_to_async
    def save_usertext(self, message, input_data):
        return UserText.objects.create(
            message=message,
            content=input_data,
        )

    #     # Save topic distributions
    #     for topic, weight in topics:
    #         TopicDistribution.objects.create(
    #             user_text=user_text,
    #             topic=topic,
    #             weight=weight
    #         )

    #     return user_text

    @database_sync_to_async
    def save_message(self, conversation, content_type, is_from_user=True, in_reply_to=None):
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )
