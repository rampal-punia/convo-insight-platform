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
    def save_message(self, conversation, content_type, is_from_user=True, in_reply_to=None):
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )


# class OrderConsumer:
#     async def _get_customer_info(self, user_id: str) -> dict:
#         """Fetch customer information"""
#         try:
#             # Assuming you have a customer model or profile
#             customer = await database_sync_to_async(Customer.objects.get)(user_id=user_id)
#             return {
#                 "user_id": str(customer.id),
#                 "name": customer.get_full_name(),
#                 "email": customer.email,
#                 "preferences": customer.preferences if hasattr(customer, 'preferences') else {},
#                 "shipping_addresses": await self._get_shipping_addresses(customer),
#                 "recent_orders": await self._get_recent_orders(customer)
#             }
#         except Customer.DoesNotExist:
#             logger.warning(f"Customer info not found for user_id: {user_id}")
#             return {}
#         except Exception as e:
#             logger.error(f"Error fetching customer info: {str(e)}")
#             return {}

#     async def _get_shipping_addresses(self, customer) -> list:
#         """Fetch customer's shipping addresses"""
#         try:
#             addresses = await database_sync_to_async(list)(
#                 customer.addresses.filter(
#                     is_active=True).order_by('-is_default')
#             )
#             return [
#                 {
#                     "id": str(addr.id),
#                     "type": addr.type,
#                     "is_default": addr.is_default,
#                     "full_address": addr.get_full_address(),
#                 }
#                 for addr in addresses
#             ]
#         except Exception as e:
#             logger.error(f"Error fetching shipping addresses: {str(e)}")
#             return []

#     async def _get_recent_orders(self, customer, limit: int = 5) -> list:
#         """Fetch customer's recent orders"""
#         try:
#             recent_orders = await database_sync_to_async(list)(
#                 customer.orders.order_by('-created')[:limit]
#             )
#             return [
#                 {
#                     "order_id": str(order.id),
#                     "status": order.get_status_display(),
#                     "total_amount": str(order.total_amount),
#                     "created_at": order.created.isoformat()
#                 }
#                 for order in recent_orders
#             ]
#         except Exception as e:
#             logger.error(f"Error fetching recent orders: {str(e)}")
#             return []

#     async def _get_cart_info(self, order_id: str) -> dict:
#         """Fetch current cart information"""
#         try:
#             order = await database_sync_to_async(Order.objects.get)(id=order_id)
#             cart_items = await database_sync_to_async(list)(order.items.select_related('product'))

#             return {
#                 "items": [
#                     {
#                         "product_id": str(item.product.id),
#                         "name": item.product.name,
#                         "quantity": item.quantity,
#                         "price": str(item.price),
#                         "total": str(item.price * item.quantity),
#                         "stock_available": item.product.stock,
#                         # Pending or Processing
#                         "modifiable": order.status in ['PE', 'PR']
#                     }
#                     for item in cart_items
#                 ],
#                 "summary": {
#                     "total_items": sum(item.quantity for item in cart_items),
#                     "subtotal": str(sum(item.price * item.quantity for item in cart_items)),
#                     "shipping": str(order.shipping_cost if hasattr(order, 'shipping_cost') else 0),
#                     "tax": str(order.tax if hasattr(order, 'tax') else 0),
#                     "total": str(order.total_amount)
#                 },
#                 "modifiable": order.status in ['PE', 'PR']
#             }
#         except Order.DoesNotExist:
#             logger.warning(f"Cart not found for order_id: {order_id}")
#             return {"items": [], "summary": {}, "modifiable": False}
#         except Exception as e:
#             logger.error(f"Error fetching cart info: {str(e)}")
#             return {"items": [], "summary": {}, "modifiable": False}

#     async def handle_intent(self, intent: str, conversation_id: str, order_id: str):
#         """Handle incoming intent and initialize the graph with comprehensive state"""
#         try:
#             # Get order details
#             order_details = await self.db_ops.get_order_details(order_id)
#             if 'error' in order_details:
#                 logger.error(
#                     f"Failed to get order details: {order_details['error']}")
#                 return

#             # Get customer info
#             customer_info = await self._get_customer_info(self.user.id)

#             # Get cart info
#             cart_info = await self._get_cart_info(order_id)

#             # Create graph configuration
#             config = GraphConfig(
#                 llm=self.llm,
#                 intent=intent,
#                 order_details=order_details,
#                 tools=self.tools,
#                 conversation_id=conversation_id
#             )

#             # Initialize graph builder
#             logger.info(f"Initializing graph builder for intent: {intent}")
#             graph_builder = GraphBuilder(config)
#             self.graph = graph_builder.build()

#             # Initialize comprehensive state
#             self.state = {
#                 "messages": await self.db_ops.get_conversation_messages(conversation_id),
#                 "customer_info": customer_info,
#                 "order_info": order_details,
#                 "cart": cart_info,
#                 "intent": intent,
#                 "conversation_id": conversation_id,
#                 "modified": False,
#                 "confirmation_pending": False,
#                 "metadata": {
#                     "last_updated": datetime.now().isoformat(),
#                     "session_id": str(uuid.uuid4()),
#                     "platform": self.platform if hasattr(self, 'platform') else 'web',
#                     "locale": self.user.preferences.get('locale', 'en_US') if hasattr(self.user, 'preferences') else 'en_US'
#                 }
#             }

#             logger.info(
#                 f"Intent handler initialized for conversation: {conversation_id}",
#                 extra={
#                     "intent": intent,
#                     "customer_id": self.user.id,
#                     "order_id": order_id,
#                     "state_size": len(str(self.state))
#                 }
#             )

#         except Exception as e:
#             logger.error(
#                 f"Error in handle_intent: {str(e)}",
#                 extra={
#                     "intent": intent,
#                     "conversation_id": conversation_id,
#                     "order_id": order_id
#                 },
#                 exc_info=True
#             )
#             raise
