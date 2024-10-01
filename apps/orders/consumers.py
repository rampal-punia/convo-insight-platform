import json

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from convochat.utils.text_chat_handler import TextChatHandler
from convochat.models import Conversation, Message, UserText, AIText
from .models import Order, OrderConversationLink, OrderItem
# from .tasks import process_ai_response, process_user_message, analyze_conversation_sentiment


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
        input_data, conversation, user_message = await self.create_conversation_db(text_data)

        # Process the message in real-time
        # analyze_topics.delay(conversation.id)
        # recognize_intent.delay(user_message.id)
        # analyze_sentiment.delay(user_message.id)

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
        conversation_id = data.get('uuid')
        print('*'*40)
        print("conversation_id is : ", conversation_id)
        order_id = data.get('order_id')

        conversation = await self.get_or_create_conversation(conversation_id, order_id)
        user_message = await self.save_message(conversation, input_type, is_from_user=True)
        await self.save_usertext(user_message, input_data)
        return input_data, conversation, user_message

    @database_sync_to_async
    def get_or_create_conversation(self, conversation_id, order_id):
        conversation, created = Conversation.objects.update_or_create(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )
        order = Order.objects.get(order_id)
        OrderConversationLink.objects.get_or_create(
            order=order,
            conversation=conversation
        )
        if created:
            Conversation.objects.filter(user=self.user, status='AC').exclude(
                id=conversation_id
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
    def save_message(self, conversation, content_type, is_from_user=True, in_reply_to=None):
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )
