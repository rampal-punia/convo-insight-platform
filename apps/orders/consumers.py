import json
from typing import Dict
from decimal import Decimal
import logging

import traceback
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from django.conf import settings

from convochat.models import Conversation
from .utils.tool_manager import (
    tool_manager,
    BaseOrderSchema,
    ModifyOrderQuantity,
    CancelOrder,
    TrackOrder,
    GetSupportInfo
)
from .utils.db_utils import DatabaseOperations
from .utils.graph_builder import GraphBuilder, GraphConfig

logger = logging.getLogger('orders')  # Get the orders logger


class OrderSupportConsumer(AsyncWebsocketConsumer):
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
        self.create_tools = tool_manager(self.db_ops)
        self.message_count = 0
        self.summary_threshold = 5

        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')
        await self.accept()

        logger.info(
            f"WebSocket connected for conversation {self.conversation_id}")

        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! You are now connected to Order Support."
        }))

        self.llm = ChatOpenAI(
            model=settings.GPT_MINI,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
            request_timeout=settings.REQUEST_GPT_TIMEOUT,
        )

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            logger.info(f"Received message type: {message_type}\n")
            logger.debug(f"Full message data: {data}")

            # Increment message count
            self.message_count += 1

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

            logger.info(f"Processing intent: {intent} for order: {order_id}\n")

            # Get conversation and order details
            conversation, order = await self.db_ops.get_or_create_conversation(conversation_id, order_id)
            # Store the intent
            conversation.current_intent = intent
            await database_sync_to_async(conversation.save)()

            # Get order details and conversation messages from the database
            order_details = await self.db_ops.get_order_details(order_id)
            conversation_messages = await self.db_ops.get_conversation_messages(conversation.id)

            # Initialize tools and graph
            tools = self.create_tools()
            config = GraphConfig(
                llm=self.llm,
                intent=intent,
                order_details=order_details,
                tools=tools,
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
                "messages": conversation_messages + [
                    HumanMessage(
                        content=f"I need help with {intent} for order #{order_id}.")
                ],
                "intent": intent,
                "order_info": order_details,
                "conversation_id": str(conversation.id)
            }

            logger.debug(f"Initial state: {initial_state}")

            # Process through graph
            async for event in self.graph.astream(initial_state, config=settings.GRAPH_CONFIG):
                logger.debug(f"Graph event received: {event}")

                # Handle nested event structure
                if isinstance(event, dict):
                    messages = None
                    if "messages" in event:
                        messages = event["messages"]
                    elif "agent" in event and "messages" in event["agent"]:
                        messages = event["agent"]["messages"]

                    if not messages:
                        logger.warning(
                            f"Could not find messages in event: {event}")
                        continue

                    message = messages[-1]
                    logger.info(f"Processing message: {message}\n")

                    if isinstance(message, AIMessage):
                        content = message.content
                        logger.info(f"AI Message content: {content}\n")

                        # Save the AI response
                        ai_message = await self.db_ops.save_message(
                            conversation=conversation,
                            content_type='TE',
                            is_from_user=False,
                            in_reply_to=user_message
                        )
                        await self.db_ops.save_aitext(ai_message, content)

                        # Check for tool calls
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            await self.send(text_data=json.dumps({
                                'type': 'confirmation_required',
                                'action': content,
                                'tool_calls': [
                                    {"name": tc["name"], "args": tc["args"]}
                                    for tc in message.tool_calls
                                ]
                            }))
                        else:
                            # Send regular response
                            await self.send(text_data=json.dumps({
                                'type': 'agent_response',
                                'message': content
                            }))

                    elif isinstance(message, ToolMessage):
                        logger.info(f"Tool Message received: {message}\n")
                        # Send tool message to client if needed
                        await self.send(text_data=json.dumps({
                            'type': 'tool_response',
                            'message': message.content
                        }))
                    else:
                        logger.warning(
                            f"Unknown message type: {type(message)}")

        except Exception as e:
            logger.error(f"Error in handle_intent: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing intent: {str(e)}"
            }))

    async def handle_message(self, data):
        """Handle ongoing conversation"""
        try:
            conversation = await database_sync_to_async(Conversation.objects.get)(id=self.conversation_id)
            conversation_messages = await self.db_ops.get_conversation_messages(conversation.id)
            user_input = data.get('message')
            order_id = data.get('order_id')

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

            # Use the existing intent from the conversation
            current_intent = conversation.current_intent or "general_inquiry"

            # Continue conversation with full context
            current_state = {
                "messages": conversation_messages + [HumanMessage(content=user_input)],
                "order_info": order_details,  # Add order info to state
                "conversation_id": str(conversation.id),
                "intent": current_intent
            }

            logger.debug(f"Current state for processing: {current_state}")

            async for event in self.graph.astream(current_state, config=settings.GRAPH_CONFIG):
                if event and ("messages" in event or "agent" in event):
                    # Handle nested event structure
                    messages = None
                    if "messages" in event:
                        messages = event["messages"]
                    elif "agent" in event and "messages" in event["agent"]:
                        messages = event["agent"]["messages"]

                    if not messages:
                        logger.warning(f"No messages found in event: {event}")
                        continue

                    message = messages[-1]
                    logger.info(f"Processing message: {message}\n")

                    # Save the AI response with proper tool calls handling
                    ai_message = await self.db_ops.save_message(
                        conversation=conversation,
                        content_type='TE',
                        is_from_user=False,
                        in_reply_to=user_message
                    )

                    # Extract tool calls if they exist
                    tool_calls = getattr(message, 'tool_calls', [])
                    if tool_calls:
                        await self.db_ops.save_aitext(ai_message, message.content, tool_calls)
                        await self.send(text_data=json.dumps({
                            'type': 'confirmation_required',
                            'action': message.content,
                            'tool_calls': [
                                {"name": tc["name"], "args": tc["args"]}
                                for tc in tool_calls
                            ]
                        }))
                    else:
                        await self.db_ops.save_aitext(ai_message, message.content)
                        await self.send(text_data=json.dumps({
                            'type': 'agent_response',
                            'message': message.content
                        }))

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
            conversation_messages = await self.db_ops.get_conversation_messages(conversation.id)

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
            print("*"*40)
            print("order_details: ", order_details)
            print("*"*40)
            tool_name = tool_call.get('name')
            tool_args = json.loads(tool_call.get('args')) if isinstance(
                tool_call.get('args'), str) else tool_call.get('args')

            logger.info(
                f"Executing tool: {tool_name} with args: {tool_args}\n")

            # Get the tool from tool manager
            tools = self.create_tools()
            tool = next((t for t in tools if t.name == tool_name), None)

            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Construct input dictionary for each tool
            tool_input = {}

            if tool_name == 'get_order_details':
                tool_input = {
                    'order_id': str(order_details['order_id']),
                    # Use actual user ID instead of 'self'
                    'customer_id': str(self.user.id)
                }

            elif tool_name == 'modify_order_quantity':
                tool_input = {
                    'order_id': str(order_details['order_id']),
                    'customer_id': str(self.user.id),
                    'product_id': tool_args.get('product_id'),
                    'new_quantity': tool_args.get('new_quantity')
                }

            elif tool_name == 'cancel_order':
                tool_input = {
                    'order_id': str(order_details['order_id']),
                    'customer_id': str(self.user.id),
                    'reason': tool_args.get('reason', 'User requested cancellation')
                }

            elif tool_name == 'track_order':
                tool_input = {
                    'order_id': str(order_details['order_id']),
                    'customer_id': str(self.user.id)
                }

            elif tool_name == 'get_support_info':
                tool_input = {
                    'order_id': str(order_details['order_id']),
                    'customer_id': str(self.user.id)
                }

            else:
                raise ValueError(f"Unknown tool operation: {tool_name}")

            logger.info(f"Prepared tool input: {tool_input}")
            result = await tool.ainvoke(tool_input)
            logger.info(f"Tool execution result: {result}\n")
            return result

        except Exception as e:
            logger.error(f"Error executing tool action: {str(e)}")
            logger.error(traceback.format_exc())
            raise
