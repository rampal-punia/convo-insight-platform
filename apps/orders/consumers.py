import json
from decimal import Decimal
import logging

import traceback
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from django.conf import settings

from convochat.models import Conversation
from .utils.tool_manager import create_tool_manager
from .utils.db_utils import DatabaseOperations
from .utils.graph_builder import GraphBuilder, GraphConfig

logger = logging.getLogger('orders')  # Get the orders logger


class OrderSupportConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.awaiting_confirmation = False
        self.pending_tool_calls = None

    async def connect(self):
        logger.info(
            f"WebSocket connection attempt from user {self.scope['user']}")

        """Initialize the consumer and database operations"""
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            logger.warning(f"Unauthenticated connection attempt")
            await self.close()
            return

        self.db_ops = DatabaseOperations(self.user)
        self.tool_manager = create_tool_manager(self.db_ops)

        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')
        await self.accept()

        logger.info(
            f"WebSocket connected for conversation {self.conversation_id}")

        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! You are now connected to Order Support."
        }))

        # chatmodel = HuggingFaceEndpoint(
        #     repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     task='chat',
        #     huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
        #     max_new_tokens=512,
        #     temperature=0,
        #     do_sample=False,
        #     repetition_penalty=1.03,
        # )
        # self.llm = ChatHuggingFace(llm=chatmodel)

        self.llm = ChatOpenAI(
            model=settings.GPT_MINI,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
            request_timeout=settings.REQUEST_GPT_TIMEOUT,
        )

        self.current_messages = []  # Add this to track current conversation
        self.graph = None

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected: {self.channel_name}")
        pass

    async def receive(self, text_data=None):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            logger.info(f"Received message type: {message_type}\n")
            logger.debug(f"Full message data: {data}")

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
        """Handle intent selection"""
        try:
            intent = data.get('intent')
            order_id = data.get('order_id')
            conversation_id = data.get('uuid')

            # Reset message history when starting new intent
            self.current_messages = []

            logger.info(f"Processing intent: {intent} for order: {order_id}\n")

            # Get conversation and order details
            conversation, order = await self.db_ops.get_or_create_conversation(conversation_id, order_id)

            # Only get initial message for the intent
            initial_message = HumanMessage(
                content=f"I need help with {intent} for order #{order_id}.")
            self.current_messages = [initial_message]

            # Store the intent
            conversation.current_intent = intent
            await database_sync_to_async(conversation.save)()

            # Get order details and conversation messages
            order_details = await self.db_ops.get_order_details(order_id)

            # For modify intent, check status immediately
            if intent == 'modify_order':
                if order_details['status'] not in ['Pending', 'Processing']:
                    status_message = f"""I notice that your order is currently in '{order_details['status']}' status. 
                    Orders can only be modified while they are in 'Pending' or 'Processing' status.
                    
                    Here's what you can do with your order in its current status:
                    """

                    status_actions = {
                        'Shipped': [
                            "Track your shipment",
                            "Contact support for delivery updates"
                        ],
                        'Delivered': [
                            "Initiate a return (if within 30 days)",
                            "Leave a review"
                        ],
                        'Cancelled': [
                            "Place a new order",
                            "View similar products"
                        ],
                        'Returned': [
                            "Check refund status",
                            "Place a new order"
                        ],
                        'In Transit': [
                            "Track your shipment",
                            "Update delivery preferences"
                        ]
                    }

                    actions = status_actions.get(
                        order_details['status'],
                        ["Contact customer support for assistance"]
                    )
                    status_message += "\n" + \
                        "\n".join(f"- {action}" for action in actions)

                    await self.send(text_data=json.dumps({
                        'type': 'agent_response',
                        'message': status_message
                    }))
                    return

            # Only get initial message for the intent if status check passed
            initial_message = HumanMessage(
                content=f"I need help with {intent} for order #{order_id}.")
            self.current_messages = [initial_message]

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

            # Create initial message
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

    async def process_graph_event(self, event, conversation, user_message):
        try:
            messages = []
            if isinstance(event, dict):
                logger.debug(f"Processing event dict: {event}")
                if "messages" in event:
                    messages = event["messages"]
                elif "agent" in event and "messages" in event["agent"]:
                    messages = event["agent"]["messages"]

                if not messages:
                    return

                for message in messages:
                    if not message:
                        continue

                    logger.info(f"Processing message type: {type(message)}")
                    logger.debug(f"Message content: {message}")

                    if isinstance(message, AIMessage):
                        content = message.content
                        logger.info(f"AI Message content: {content}")

                        # Log tool calls if present
                        if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                            tool_calls = message.additional_kwargs['tool_calls']
                            logger.info(f"Tool calls found: {tool_calls}")

                            # Process each tool call
                            for tool_call in tool_calls:
                                if isinstance(tool_call, dict) and 'function' in tool_call:
                                    tool_name = tool_call['function']['name']
                                    logger.info(
                                        f"Processing tool: {tool_name}")

                                    if self.tool_manager.is_sensitive_tool(tool_name):
                                        self.awaiting_confirmation = True
                                        self.pending_tool_calls = tool_calls
                                        logger.info(
                                            f"Sending confirmation for sensitive tool: {tool_name}")
                                        await self.send(text_data=json.dumps({
                                            'type': 'confirmation_required',
                                            'action': content or f"Confirm {tool_name} operation?",
                                            'tool_calls': tool_calls
                                        }))
                                        return

                        # Save the AI response
                        ai_message = await self.db_ops.save_message(
                            conversation=conversation,
                            content_type='TE',
                            is_from_user=False,
                            in_reply_to=user_message
                        )
                        await self.db_ops.save_aitext(ai_message, content)

                        # Only save and send message if it has content
                        await self.send(text_data=json.dumps({
                            'type': 'agent_response',
                            'message': content
                        }))

        except Exception as e:
            logger.error(f"Error processing graph event: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing response: {str(e)}"
            }))

    async def handle_message(self, data):
        """Handle ongoing conversation"""
        try:
            conversation = await database_sync_to_async(Conversation.objects.get)(id=self.conversation_id)
            user_input = data.get('message')
            order_id = data.get('order_id')

            # Get existing conversation history first
            conversation_history = await self.db_ops.get_conversation_history(self.conversation_id)

            # Add new user message
            self.current_messages = conversation_history + \
                [HumanMessage(content=user_input)]

            # Get order details for context
            order_details = await self.db_ops.get_order_details(order_id)

            logger.info(
                f"Processing message for order {order_id}: {user_input}")

            # Save user message
            user_message = await self.db_ops.save_message(
                conversation=conversation,
                content_type='TE',
                is_from_user=True
            )
            await self.db_ops.save_usertext(user_message, user_input)

            # Check if we're awaiting confirmation
            if self.awaiting_confirmation and any(word in user_input.lower() for word in ['yes', 'confirm', 'proceed']):
                if self.pending_tool_calls:
                    await self.handle_confirmation({
                        'approved': True,
                        'tool_calls': self.pending_tool_calls,
                        'uuid': self.conversation_id
                    })
                    return

            # Continue conversation with full context
            current_state = {
                "messages": self.current_messages,
                "order_info": order_details,  # Add order info to state
                "conversation_id": self.conversation_id,
                "intent": conversation.current_intent
            }

            logger.debug(f"Current state for processing: {current_state}")

            async for event in self.graph.astream(current_state, config=settings.GRAPH_CONFIG):
                if event and ("messages" in event or "agent" in event):
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
            conversation_id = data.get('uuid')

            logger.info(
                f"Processing confirmation: approved={approved}, tool_calls={tool_calls}")

            # Get conversation and related data
            conversation = await database_sync_to_async(Conversation.objects.get)(id=conversation_id)

            # Get the associated order from the tool calls
            order_id = None
            if tool_calls and len(tool_calls) > 0 and 'args' in tool_calls[0]:
                order_id = tool_calls[0]['args'].get('order_id')

            if not order_id:
                # Try to get order_id from the conversation link
                order_link = await database_sync_to_async(lambda: conversation.order_links.first())()
                if order_link:
                    order_id = await database_sync_to_async(lambda: order_link.order.id)()

            if not order_id:
                raise ValueError(
                    "Could not determine order ID for confirmation")

            # Get order details
            order_details = await self.db_ops.get_order_details(order_id)

            if approved:
                logger.info("User approved action, proceeding with tools")

                # Execute the approved action using tool
                result = await self.execute_tool_action(tool_calls[0], order_details)

                # Update the conversation with the result
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

                # Send completion notification with updated card
                await self.send(text_data=json.dumps({
                    'type': 'operation_complete',
                    'message': result,
                    'update_elements': [
                        {
                            'selector': f'.card[data-order-id="{order_details["order_id"]}"]',
                            'html': await self.db_ops.render_order_card(order_details)
                        }
                    ],
                    'completion_message': 'Order successfully updated!'
                }, default=self.decimal_default))

            else:
                decline_message = ToolMessage(
                    content=f"Action denied by user. Reason: {reason}",
                    tool_call_id=tool_calls[0].get(
                        'id', 'unknown') if tool_calls else 'unknown'
                )
                logger.info(
                    f"User declined action. Reason: {decline_message}\n")

                # Save the decline message
                ai_message = await self.db_ops.save_message(
                    conversation=conversation,
                    content_type='TE',
                    is_from_user=False
                )
                await self.db_ops.save_aitext(ai_message, decline_message.content)

                # Reset confirmation state
                self.awaiting_confirmation = False
                self.pending_tool_calls = None

                # Send decline notification
                await self.send(text_data=json.dumps({
                    'type': 'agent_response',
                    'message': decline_message
                }, default=self.decimal_default))

        except Exception as e:
            logger.error(f"Error in handle_confirmation: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing confirmation: {str(e)}"
            }, default=self.decimal_default))

    def decimal_default(self, obj):
        """Handle Decimal serialization"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def execute_tool_action(self, tool_call, order_details):
        """Execute the approved tool action using the tool manager"""
        try:
            # Extract tool info from the function field
            if not tool_call or not isinstance(tool_call, dict):
                raise ValueError("Invalid tool call format")

            # Get function details
            function_info = tool_call.get('function', {})
            tool_name = function_info.get('name')

            if not tool_name:
                logger.error(f"No tool name found in tool call: {tool_call}")
                raise ValueError("Tool name not found in tool call")

            # Parse arguments properly
            try:
                tool_args = json.loads(function_info.get('arguments', '{}')) if isinstance(
                    function_info.get('arguments'), str) else function_info.get('arguments', {})
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing tool arguments: {e}")
                raise ValueError(f"Invalid tool arguments format: {e}")

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Get the tool from tool manager
            all_tools = self.tool_manager.get_all_tools()
            tool = next((t for t in all_tools if t.name == tool_name), None)

            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Construct input dictionary for each tool
            tool_input = {}

            if tool_name == 'modify_order_quantity':
                can_modify, message = await self.db_ops.validate_order_status_for_modification(
                    tool_args.get('order_id')
                )
                if not can_modify:
                    return message

                tool_input = {
                    'order_id': str(tool_args.get('order_id')),
                    'customer_id': str(tool_args.get('customer_id')),
                    'product_id': int(tool_args.get('product_id')),
                    'new_quantity': int(tool_args.get('new_quantity'))
                }

            elif tool_name == 'cancel_order':
                tool_input = {
                    'order_id': str(tool_args.get('order_id')),
                    'customer_id': str(tool_args.get('customer_id')),
                    'reason': tool_args.get('reason', 'User requested cancellation')
                }

            else:
                raise ValueError(f"Unknown tool operation: {tool_name}")

            logger.info(f"Prepared tool input: {tool_input}")

            result = await tool.ainvoke(tool_input)
            logger.info(f"Tool execution result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error executing tool action: {str(e)}")
            logger.error(traceback.format_exc())
            raise
