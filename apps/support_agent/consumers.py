"""
Intent-aware Order Support Consumer with Dynamic Routing.

This module implements a WebSocket consumer for handling order-related queries
with dynamic intent detection and contextual conversation management.
"""

import json
import logging
import traceback
from typing import Dict, Any, Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from langchain_core.messages import AIMessage, HumanMessage
from .sa_utils.tool_manager import get_all_tools
from .sa_utils.state import ECommerseState
from .sa_utils.graph_builder import graph
from .sa_utils.prompt_manager import assistant_prompt
from orders.utils.db_utils import DatabaseOperations

logger = logging.getLogger('orders')


class SupportAgentConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for handling order-related support conversations.
    Implements dynamic intent detection and contextual conversation management.
    """

    async def connect(self):
        """Initialize connection and required components"""
        try:
            # Initialize components
            self.user = self.scope["user"]
            self.user_id = self.user.id
            self.conversation_id = self.scope['url_route']['kwargs'].get(
                'conversation_id')

            logger.info(
                f'''WebSocket connection attempt from: 
                ***********
                User: {self.user}
                Conversation ID: {self.conversation_id}
                ***********'''
            )

            # Validate authentication and accept
            if not self.scope["user"].is_authenticated:
                logger.warning("Unauthenticated connection attempt")
                await self.close()
                return
            await self.accept()

            # "user_id": self.user_id,
            # "conversation_id": self.conversation_id,
            # "db_ops": DatabaseOperations(self.user)
            self.tools = get_all_tools()
            self.username_cap = self.user.username.capitalize()

            # Send welcome message
            await self.send_json({
                'type': 'welcome',
                'message': f"Welcome, {self.username_cap}! How can I assist you with your order today?"
            })

        except Exception as e:
            logger.error(f"Error in connect: {str(e)}")
            logger.error(traceback.format_exc())
            await self.close()

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        pass

    async def receive(self, text_data=None):
        """Handle incoming WebSocket messages with dynamic intent detection"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'message')

            # Route message based on type
            handlers = {
                'message': self.handle_message,
                'confirmation': self.handle_confirmation
            }

            handler = handlers.get(message_type)
            if handler:
                await handler(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self.send_error("Unknown message type")

        except Exception as e:
            logger.error(f"Error in receive: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error(f"Error processing request: {str(e)}")

    async def handle_message(self, data: Dict[str, Any]):
        """Process user message with state awareness"""
        try:
            user_input = data.get('message')
            logger.info(f"Received user input: {user_input}")
            self.config = {
                "configurable": {
                    "model": settings.GPT_MINI_STRING,
                    "thread_id": self.conversation_id,
                    "max_search_results": 3,
                    "system_prompt": assistant_prompt,
                    "max_tool_retries": 2,
                    "user_info": {
                        "username": self.user.username,
                        "id": self.user.id,
                        "email": self.user.email
                    }
                }
            }

            # Get existing state or create new
            try:
                # current_state = graph.get_state(self.config)
                current_state = graph.get_state(
                    self.config).values["messages"][-1]
                logger.info(f"Current state is {current_state}")

                if current_state and current_state.values.get("messages"):
                    # Append to existing messages
                    initial_state = ECommerseState(
                        messages=[*current_state.values["messages"],
                                  HumanMessage(content=user_input)]
                    )
                    logger.info(
                        f"Appended current state to initial state: {initial_state}")
                else:
                    # Create new state
                    initial_state = ECommerseState(
                        messages=[HumanMessage(content=user_input)]
                    )
                    logger.info(
                        f"Create new state as initial state: {initial_state}")
            except Exception as e:
                logger.warning(
                    f"Could not get previous state: {e}. Starting fresh.")
                # Create new state
                initial_state = ECommerseState(
                    messages=[HumanMessage(content=user_input)]
                )
                logger.info(f"Initial state: {initial_state}")

            # Track seen message IDs to prevent duplicates
            seen_messages = set()

            async for event in graph.astream(
                initial_state,
                self.config,
                stream_mode="values"
            ):
                if "messages" in event:
                    # Get only the latest message
                    messages = event["messages"]

                    logger.info(
                        f'''event["messages"](message) : {messages}
                        Type of messages: {type(messages)}
                        '''
                    )
                    if not isinstance(messages, list):
                        messages = [messages]

                    latest_message = messages[-1]
                    # Skip if we've seen this message before
                    if hasattr(latest_message, 'id') and latest_message.id in seen_messages:
                        continue
                    if hasattr(latest_message, 'id'):
                        seen_messages.add(latest_message.id)

                    # Skip human messages
                    if isinstance(latest_message, HumanMessage):
                        continue
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        # Tool execution in progress
                        tool_call = latest_message.tool_calls[0]
                        await self.send_json({
                            "type": "agent_response",
                            "message": "Let me process that for you...",
                            "tool_call": {
                                "name": tool_call["name"],
                                "args": tool_call["args"],
                                "id": tool_call["id"]
                            }
                        })
                    else:
                        # Regular message response
                        await self.send_json({
                            "type": "agent_response",
                            "message": latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                        })
                    logger.debug(f"Processed message: {latest_message}")
            # Signal completion
            await self.send_json({
                "type": "processing_complete"
            })

        except Exception as e:
            logger.error(f"Error in handle_message: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error(f"Error processing request: {str(e)}")

    async def send_json(self, data: Dict[str, Any]):
        """Send JSON response to WebSocket"""
        await self.send(text_data=json.dumps(data))

    async def send_error(self, message: str):
        """Send error message and hide spinner"""
        await self.send_json({
            'type': 'error',
            'message': message
        })

    def format_order_list(self, orders: list) -> str:
        """Format the list of orders for display"""
        formatted_orders = []
        for order in orders:
            formatted_orders.append(
                f"Order #{order['id']} - Placed on {order['created_date']}\n"
                f"Status: {order['status']}\n"
                f"Items: {order['item_count']}\n"
                f"Total: ${order['total_amount']}\n"
            )
        return "\n".join(formatted_orders)

    async def handle_confirmation_request(self, message: AIMessage, conversation: Any):
        """Handle requests requiring user confirmation"""
        try:
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            if not tool_calls:
                return

            tool_call = tool_calls[0]
            args = json.loads(tool_call['args'])

            # Format confirmation message
            confirmation_message = self.format_confirmation_message(
                tool_call['name'],
                args
            )

            await self.send_json({
                'type': 'confirmation_required',
                'message': confirmation_message,
                'tool_calls': tool_calls
            })

        except Exception as e:
            logger.error(f"Error handling confirmation request: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error("Error processing confirmation request")

    async def handle_confirmation(self, data: Dict[str, Any]):
        """Handle tool confirmation responses"""
        try:
            approved = data.get('approved')
            tool_calls = data.get('tool_calls', [])
            reason = data.get('reason', '')

            if not tool_calls:
                await self.send_error("No tool calls provided for confirmation")
                return

            # Execute confirmed tool action
            tool_call = tool_calls[0]
            if approved:
                result = await self.execute_tool_action(tool_call)
                if result:
                    await self.send_json({
                        'type': 'operation_complete',
                        'message': result
                    })
            else:
                await self.send_json({
                    'type': 'agent_response',
                    'message': f"Action declined. Reason: {reason}"
                })

        except Exception as e:
            logger.error(f"Error handling confirmation: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error("Error processing confirmation")

    async def execute_tool_action(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """Execute confirmed tool action"""
        try:
            function_info = tool_call.get('name', {})
            tool_name = function_info.get('name')
            tool_args = json.loads(function_info.get('args', '{}'))

            # Get tool from available tools
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                return f"Tool {tool_name} not found"

            # Execute tool
            result = await tool.ainvoke({**tool_args, 'db_ops': self.db_ops})
            return result

        except Exception as e:
            logger.error(f"Error executing tool action: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def needs_confirmation(message: AIMessage) -> bool:
        """Determine if a message needs user confirmation"""
        if not message.additional_kwargs.get('tool_calls'):
            return False

        tool_name = message.additional_kwargs['tool_calls'][0]['name']
        return tool_name in {'modify_order_quantity', 'cancel_order'}

    def format_confirmation_message(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Format confirmation message based on tool and arguments"""
        templates = {
            'modify_order_quantity': (
                f"Please confirm that you want to change the quantity to "
                f"{args.get('new_quantity')} for Order #{args.get('order_id')}"
            ),
            'cancel_order': (
                f"Please confirm that you want to cancel Order #{args.get('order_id')}. "
                f"Reason: {args.get('reason', 'No reason provided')}"
            )
        }
        return templates.get(
            tool_name,
            f"Please confirm this action: {tool_name} with {args}"
        )
