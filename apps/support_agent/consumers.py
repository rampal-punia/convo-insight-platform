import json
import logging
import traceback
from typing import Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from convochat.models import Conversation
from .utils.tool_manager import create_tool_manager
from orders.utils.db_utils import DatabaseOperations
from .utils.graph_builder import GraphBuilder, GraphConfig

logger = logging.getLogger('orders')


class OrderSupportConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.awaiting_confirmation = False
        self.pending_tool_calls = None

    async def connect(self):
        """Initialize consumer and establish WebSocket connection"""
        logger.info(
            f"WebSocket connection attempt from user {self.scope['user']}")

        # Initialize user and check authentication
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            logger.warning("Unauthenticated connection attempt")
            await self.close()
            return

        # Initialize database operations and tool manager
        self.db_ops = DatabaseOperations(self.user)
        self.tool_manager = create_tool_manager(self.db_ops)

        # Get conversation ID from URL route
        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')
        await self.accept()

        logger.info(
            f"WebSocket connected for conversation {self.conversation_id}")

        # Send welcome message
        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! You are now connected to Order Support."
        }))

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.GPT_MINI,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
            streaming=True
        )

        self.current_messages = []
        self.graph = None

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        logger.info(f"WebSocket disconnected: {self.channel_name}")
        pass

    async def receive(self, text_data=None):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            logger.info(f"Received message type: {message_type}")
            logger.debug(f"Full message data: {data}")

            # Route message based on type
            if message_type == 'intent':
                await self.handle_intent(data)
            elif message_type == 'message':
                await self.handle_message(data)
            elif message_type == 'confirmation':
                await self.handle_confirmation(data)
            else:
                logger.warning(
                    f"Unknown message type received: {message_type}")
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Unknown message type'
                }))

        except Exception as e:
            logger.error(f"Error in receive: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing request: {str(e)}"
            }))

    async def handle_intent(self, data):
        """Handle intent selection and initialization"""
        try:
            intent = data.get('intent')
            order_id = data.get('order_id')
            logger.info(f"Processing intent: {intent} for order: {order_id}")
            conversation_id = data.get('uuid')

            # Reset message history for new intent
            self.current_messages = []

            logger.debug(f"Full intent data: {data}")

            # Check if intent requires sensitive tools
            is_sensitive = self.tool_manager.is_sensitive_tool(intent)
            logger.debug(f"Intent sensitivity check: {is_sensitive}")

            # Get conversation and order details
            conversation, order = await self.db_ops.get_or_create_conversation(
                conversation_id,
                order_id
            )

            # Create initial message for intent
            initial_message = HumanMessage(
                content=f"I need help with {intent} for order #{order_id}."
            )
            self.current_messages = [initial_message]

            # Store intent in conversation
            conversation.current_intent = intent
            await database_sync_to_async(conversation.save)()

            # Get order details
            order_details = await self.db_ops.get_order_details(order_id)

            # Add status check logging
            order_details = await self.db_ops.get_order_details(order_id)
            logger.debug(f"Order status: {order_details['status']}")

            # For modify_order intent, check status immediately
            if intent == 'modify_order_quantity':
                can_modify = order_details['status'] in [
                    'Pending', 'Processing']
                logger.info(
                    f"Can modify order: {can_modify}, Message: {data}")
                if not can_modify:
                    await self.send(text_data=json.dumps({
                        'type': 'error',
                        'message': f"Cannot modify order in {order_details['status']} status. Only pending or processing orders can be modified."
                    }))
                    return

            # Initialize graph with tool manager
            config = GraphConfig(
                llm=self.llm,
                intent=intent,
                order_details=order_details,
                tool_manager=self.tool_manager,
                conversation_id=conversation_id
            )

            graph_builder = GraphBuilder(config)
            self.graph = graph_builder.build()

            # Save the graph visualization
            # output_path = await graph_builder.save_graph_visualization()
            # print(f"Graph saved to: {output_path}")

            # # Use the png_path as needed in your WebSocket communication
            # relative_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
            # media_url = os.path.join(settings.MEDIA_URL, relative_path)

            # await self.send(text_data=json.dumps({
            #     'type': 'graph_visualization',
            #     'url': media_url
            # }))

            # Save initial message
            user_message = await self.db_ops.save_message(
                conversation=conversation,
                content_type='TE',
                is_from_user=True
            )
            await self.db_ops.save_usertext(user_message, f"Request for {intent}")

            # Initialize state
            initial_state = {
                "messages": self.current_messages,
                "intent": intent,
                "order_info": order_details,
                "conversation_id": str(conversation.id),
                "confirmation_pending": False,
                "completed": False
            }

            logger.debug(f"Initial state: {initial_state}")

            # Process through graph
            async for event in self.graph.astream(initial_state, config=settings.GRAPH_CONFIG):
                if event.get("error"):
                    await self.send(text_data=json.dumps({
                        'type': 'error',
                        'message': event["error"]
                    }))
                    return

                await self.process_graph_event(event, conversation, None)

        except Exception as e:
            logger.error(f"Error in handle_intent: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing intent: {str(e)}"
            }))

    async def handle_message(self, data):
        """Handle ongoing conversation messages"""
        try:
            conversation = await database_sync_to_async(lambda: Conversation.objects.get(id=self.conversation_id))()
            user_input = data.get('message')
            order_id = data.get('order_id')

            # Get conversation history
            conversation_history = await self.db_ops.get_conversation_history(self.conversation_id)
            # Get order details
            order_details = await self.db_ops.get_order_details(order_id)

            # Add new message
            self.current_messages = conversation_history + \
                [HumanMessage(content=user_input)]

            logger.info(
                f"Processing message for order {order_id}: {user_input}")

            # Save user message
            user_message = await self.db_ops.save_message(
                conversation=conversation,
                content_type='TE',
                is_from_user=True
            )
            await self.db_ops.save_usertext(user_message, user_input)

            # Handle confirmation responses
            if self.awaiting_confirmation and any(word in user_input.lower() for word in ['yes', 'confirm', 'proceed']):
                if self.pending_tool_calls:
                    await self.handle_confirmation({
                        'approved': True,
                        'tool_calls': self.pending_tool_calls
                    })
                    return

            # Process message through graph
            current_state = {
                "messages": self.current_messages,
                "order_info": order_details,
                "conversation_id": self.conversation_id,
                "intent": conversation.current_intent
            }

            # Process through graph
            async for event in self.graph.astream(current_state, config=settings.GRAPH_CONFIG):
                await self.process_graph_event(event, conversation, user_message)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing message: {str(e)}"
            }))

    async def handle_confirmation(self, data):
        """Handle tool confirmation responses"""
        try:
            approved = data.get('approved')
            tool_calls = data.get('tool_calls', [])
            reason = data.get('reason', '')
            logger.info(f"Tool calls: {tool_calls}")

            # Get conversation and order
            conversation = await database_sync_to_async(lambda: Conversation.objects.get(id=self.conversation_id))()

            # Get order_id
            order_id = await self._get_order_id_from_conversation(conversation)
            if not order_id:
                raise ValueError(
                    "Could not determine order ID for confirmation")

            # Get order details
            order_details = await self.db_ops.get_order_details(order_id)

            if approved:
                logger.info("User approved action, proceeding with tools")

                # Execute the approved action
                result = await self.execute_tool_action(tool_calls[0], order_details)

                # Save result to conversation
                ai_message = await self.db_ops.save_message(
                    conversation=conversation,
                    content_type='TE',
                    is_from_user=False
                )
                await self.db_ops.save_aitext(
                    ai_message,
                    f"Action completed successfully: {result}",
                    tool_calls=tool_calls
                )

                # Reset confirmation state
                self.awaiting_confirmation = False
                self.pending_tool_calls = None

                # Update UI
                await self.send(text_data=json.dumps({
                    'type': 'operation_complete',
                    'message': result,
                    'update_elements': [{
                        'selector': f'.card[data-order-id="{order_details["order_id"]}"]',
                        'html': await self.db_ops.render_order_card(order_details)
                    }],
                    'completion_message': 'Order successfully updated!'
                }))

            else:
                # Handle declined action
                decline_message = f"Action declined by user. Reason: {reason}"

                # Save decline message
                ai_message = await self.db_ops.save_message(
                    conversation=conversation,
                    content_type='TE',
                    is_from_user=False
                )
                await self.db_ops.save_aitext(ai_message, decline_message)

                # Reset confirmation state
                self.awaiting_confirmation = False
                self.pending_tool_calls = None

                # Send decline notification
                await self.send(text_data=json.dumps({
                    'type': 'agent_response',
                    'message': decline_message
                }))

        except Exception as e:
            logger.error(f"Error in handle_confirmation: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing confirmation: {str(e)}"
            }))

    async def process_graph_event(self, event, conversation, user_message):
        """Process events from the graph"""
        try:
            if isinstance(event, dict):
                logger.debug(f"Processing event: {event}")

                messages = event.get("messages", [])
                if not messages and "agent" in event:
                    messages = event["agent"].get("messages", [])

                for message in messages:
                    if not message or not isinstance(message, AIMessage):
                        continue

                    content = message.content
                    logger.info(f"AI Message content: {content}")

                    # Save the message
                    ai_message = await self.db_ops.save_message(
                        conversation=conversation,
                        content_type='TE',
                        is_from_user=False,
                        in_reply_to=user_message
                    )
                    await self.db_ops.save_aitext(ai_message, content)

                    # Check if this is a confirmation request message
                    if "confirm" in content.lower() and user_message:
                        logger.info("Detected confirmation request")
                        order_id = await self._get_order_id_from_conversation(conversation)
                        new_quantity = await self.db_ops._extract_new_quantity(user_message.user_text.content)
                        product_id = await self._get_product_id_from_conversation(conversation)

                        if all([order_id, new_quantity, product_id]):
                            tool_calls = [{
                                'function': {
                                    'name': 'modify_order_quantity',
                                    'arguments': json.dumps({
                                        'order_id': order_id,
                                        'product_id': product_id,
                                        'new_quantity': new_quantity
                                    })
                                }
                            }]

                            logger.info(
                                f"Sending confirmation required with tool_calls: {tool_calls}")

                            # Send confirmation dialog
                            await self.send(text_data=json.dumps({
                                'type': 'confirmation_required',
                                'action': f"Please confirm that you want to change the quantity to {new_quantity} for Order #{order_id}",
                                'tool_calls': tool_calls
                            }))
                            return

                    # If not a confirmation request, send normal response
                    await self.send(text_data=json.dumps({
                        'type': 'agent_response',
                        'message': content
                    }))

        except Exception as e:
            logger.error(f"Error in process_graph_event: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing response: {str(e)}"
            }))

    async def _get_product_id_from_conversation(self, conversation) -> Optional[str]:
        """Get product ID from the first order item"""
        try:
            @database_sync_to_async
            def get_product_id():
                order_link = conversation.order_links.select_related(
                    'order').first()
                if order_link and order_link.order.items.exists():
                    first_item = order_link.order.items.select_related(
                        'product').first()
                    return str(first_item.product.id) if first_item else None
                return None

            product_id = await get_product_id()
            logger.info(f"Retrieved product ID: {product_id}")
            return product_id

        except Exception as e:
            logger.error(f"Error getting product ID: {str(e)}")
            return None

    async def _get_order_id_from_conversation(self, conversation) -> Optional[str]:
        """Get order ID from conversation link"""
        try:
            @database_sync_to_async
            def get_order_id():
                order_link = conversation.order_links.select_related(
                    'order').first()
                return str(order_link.order.id) if order_link else None

            order_id = await get_order_id()
            logger.info(f"Retrieved order ID: {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"Error getting order ID: {str(e)}")

    async def execute_tool_action(self, tool_call, order_details):
        """Execute approved tool actions"""
        try:
            if not tool_call or not isinstance(tool_call, dict):
                raise ValueError("Invalid tool call format")

            function_info = tool_call.get('function', {})
            tool_name = function_info.get('name')

            if not tool_name:
                raise ValueError("Tool name not found in tool call")

            # Parse arguments
            tool_args = self.parse_tool_arguments(
                function_info.get('arguments', '{}'))

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Get tool from manager
            tool = self.get_tool_from_manager(tool_name)
            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Construct tool input
            tool_input = await self.construct_tool_input(tool_name, tool_args)

            # Execute tool
            result = await tool.ainvoke(tool_input)
            logger.info(f"Tool execution result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error executing tool action: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def parse_tool_arguments(self, args_str):
        """Parse tool arguments from string or dict format"""
        try:
            if isinstance(args_str, str):
                return json.loads(args_str)
            return args_str
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tool arguments: {e}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Invalid tool arguments format: {e}")

    def get_tool_from_manager(self, tool_name: str):
        """Get tool instance from tool manager"""
        all_tools = self.tool_manager.get_all_tools()
        return next((t for t in all_tools if t.name == tool_name), None)

    async def construct_tool_input(self, tool_name: str, tool_args: dict) -> dict:
        """Construct appropriate input for each tool type"""
        if tool_name == 'modify_order_quantity':
            # Validate order status before modification
            can_modify, message = await self.db_ops.validate_order_status_for_modification(
                tool_args.get('order_id')
            )
            logger.debug(f"Can modify order: {can_modify}")
            if not can_modify:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f"Cannot modify order. Only pending or processing orders can be modified."
                }))
                return

            return {
                'order_id': str(tool_args.get('order_id')),
                'customer_id': str(tool_args.get('customer_id')),
                'product_id': int(tool_args.get('product_id')),
                'new_quantity': int(tool_args.get('new_quantity'))
            }

        elif tool_name == 'track_order':
            return {
                'order_id': str(tool_args.get('order_id')),
                'customer_id': str(tool_args.get('customer_id')),
                'include_history': bool(tool_args.get('include_history', False))
            }

        elif tool_name == 'get_tracking_details':
            return {
                'order_id': str(tool_args.get('order_id')),
                'customer_id': str(tool_args.get('customer_id')),
                'include_history': bool(tool_args.get('include_history', True))
            }

        elif tool_name == 'get_shipment_location':
            return {
                'order_id': str(tool_args.get('order_id')),
                'customer_id': str(tool_args.get('customer_id'))
            }

        elif tool_name == 'get_delivery_estimate':
            return {
                'order_id': str(tool_args.get('order_id')),
                'customer_id': str(tool_args.get('customer_id'))
            }

        elif tool_name == 'cancel_order':
            return {
                'order_id': str(tool_args.get('order_id')),
                'customer_id': str(tool_args.get('customer_id')),
                'reason': tool_args.get('reason', 'User requested cancellation')
            }

        else:
            raise ValueError(f"Unknown tool operation: {tool_name}")
