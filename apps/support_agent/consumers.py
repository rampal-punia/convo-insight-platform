"""
Intent-aware Order Support Consumer with Dynamic Routing.

This module implements a WebSocket consumer for handling order-related queries
with dynamic intent detection and contextual conversation management.
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from langchain_core.messages import AIMessage, HumanMessage
from .sa_utils.flow_manager import ConversationFlowManager
from .sa_utils.intent_router import IntentRouter
from .sa_utils.context_manager import ConversationContextManager
from .sa_utils.tool_manager import get_all_tools
from .sa_utils.graph_builder import create_customer_support_agent, State
from orders.utils.db_utils import DatabaseOperations

logger = logging.getLogger('orders')


class SupportAgentConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for handling order-related support conversations.
    Implements dynamic intent detection and contextual conversation management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_manager = None
        self.intent_router = None
        self.db_ops = None
        self.tools = None
        self.graph = None
        self.conversation_id = None
        self.current_intent = None
        self.current_order_id = None

    async def connect(self):
        """Initialize connection and required components"""
        try:
            logger.info(
                f"WebSocket connection attempt from user {self.scope['user']}")

            # Validate authentication
            if not self.scope["user"].is_authenticated:
                logger.warning("Unauthenticated connection attempt")
                await self.close()
                return

            # Initialize components
            self.user = self.scope["user"]
            self.user_id = self.user.id
            self.conversation_id = self.scope['url_route']['kwargs'].get(
                'conversation_id')
            self.conversation_state = 'initial'

            # Initialize utility classes
            self.db_ops = DatabaseOperations(self.user)
            self.tools = get_all_tools()
            self.context_manager = ConversationContextManager(self.db_ops)
            self.intent_router = IntentRouter()
            self.username_cap = self.user.username.capitalize()
            self.flow_manager = ConversationFlowManager(
                self.username_cap,
                self.user_id,
                self.conversation_id,
                self.db_ops,
                self.context_manager,
                self.intent_router
            )

            # Initialize state with basic information
            initial_state = {
                'current_intent': 'initial',
                'last_message_time': datetime.now(timezone.utc),
                'conversation_metrics': {
                    'total_messages': 0,
                    'user_messages': 0,
                    'ai_messages': 0,
                    'intent_changes': 0
                }
            }

            await self.context_manager.update_context(
                self.conversation_id,
                initial_state
            )

            await self.accept()

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
        try:
            if self.conversation_id:
                await self.context_manager.save_conversation_state(
                    self.conversation_id,
                    {
                        'last_intent': self.current_intent,
                        'status': 'disconnected'
                    }
                )
            logger.info(f"WebSocket disconnected: {self.channel_name}")

        except Exception as e:
            logger.error(f"Error in disconnect: {str(e)}")
            logger.error(traceback.format_exc())

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
            logger.info(f"Received user input: {user_input}")  # Add logging

            # Process message through flow manager
            new_state, response = await self.flow_manager.process_message(
                user_input,
                self.conversation_state
            )

            logger.info(
                f"Flow manager returned state: {new_state.state}, has_response: {response is not None}")

            # Update conversation state
            self.conversation_state = new_state.state

            # If we have a direct response, send it
            if response:
                await self.send_json({
                    'type': 'agent_response',
                    'message': response
                })
                return

            # If we're in active state, process through graph
            if new_state.state == 'active':
                logger.info(
                    f"Processing through graph. Intent: {new_state.intent}, Order ID: {new_state.order_id}")

                conversation, order = await self.db_ops.get_or_create_conversation(
                    self.conversation_id,
                    new_state.order_id
                )

                if not conversation:
                    raise ValueError("Failed to create conversation")

                # Save user message
                message = await self.db_ops.save_message(
                    conversation_id=self.conversation_id,
                    content_type='TE',
                    is_from_user=True,
                )

                if message:
                    await self.db_ops.save_usertext(message, user_input)

                    # Process through graph
                    await self.process_message(
                        conversation=conversation,
                        user_input=user_input,
                        order_id=new_state.order_id,
                        intent=new_state.intent,
                        order_info=new_state.order_info
                    )
                else:
                    raise ValueError("Failed to save message")

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error("Error processing your message")

    async def process_message(self, conversation, user_input: str, order_id: str,
                              intent: str, order_info: Dict):
        """Process message through LangGraph with context"""
        try:
            logger.info(
                f"Processing message. Intent: {intent}, Order ID: {order_id}")

            # Get conversation history
            conversation_history = await self.db_ops.get_conversation_history(
                self.conversation_id
            )

            # Prepare message state
            messages_state = {
                "messages": conversation_history + [HumanMessage(content=user_input)],
                "order_info": order_info,
                "intent": intent,
                "conversation_id": self.conversation_id,
                "confirmation_pending": False,
                "completed": False,
                "context": {
                    "order_id": order_id,
                    "current_intent": intent,
                    "last_action": None
                }
            }

            # Initialize graph with State class if needed
            if not self.graph:
                config = {
                    "llm": settings.GPT_MINI_STRING,
                    "intent": intent,
                    "order_details": order_info,
                    "tool_manager": self.tools,
                    "conversation_id": self.conversation_id
                }
                self.graph = create_customer_support_agent(config)
                logger.info("Created new graph instance")

            logger.info(f"Processing with state: {messages_state}")

            try:
                # Process through graph
                async for event in self.graph.astream(messages_state):
                    logger.info(f"Received graph event: {event}")
                    await self.handle_graph_event(event, conversation)

            except Exception as e:
                logger.error(f"Error in graph processing: {str(e)}")
                logger.error(traceback.format_exc())
                await self.send_json({
                    'type': 'error',
                    'message': "I'm having trouble processing your request. Please try again."
                })
                # Hide loading spinner
                await self.send_json({
                    'type': 'processing_complete'
                })

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_json({
                'type': 'error',
                'message': "Error processing your message"
            })
            # Hide loading spinner
            await self.send_json({
                'type': 'processing_complete'
            })

    async def handle_graph_event(self, event: Dict[str, Any], conversation: Any):
        """Handle events from the graph execution"""
        try:
            messages = []
            if isinstance(event, dict):
                # Check for messages in different possible locations
                if 'messages' in event:
                    messages = event['messages']
                elif 'assistant' in event and 'messages' in event['assistant']:
                    messages = event['assistant']['messages']

            logger.info(f"Processing messages from event: {messages}")

            for message in messages:
                if not isinstance(message, AIMessage):
                    continue

                logger.info(f"Processing AI message: {message.content}")
                if hasattr(message, 'tool_calls'):
                    logger.info(f"Tool calls present: {message.tool_calls}")

                # Save AI message
                ai_message = await self.db_ops.save_message(
                    conversation_id=self.conversation_id,
                    content_type='TE',
                    is_from_user=False,
                )

                await self.db_ops.save_aitext(
                    ai_message,
                    content=message.content,
                    tool_calls=message.additional_kwargs.get('tool_calls', [])
                )

                # Check for confirmation needs
                if self.needs_confirmation(message):
                    logger.info("Confirmation needed for message")
                    await self.handle_confirmation_request(message, conversation)
                else:
                    await self.send_json({
                        'type': 'agent_response',
                        'message': message.content
                    })

            # Always hide spinner after processing
            await self.send_json({
                'type': 'processing_complete'
            })

        except Exception as e:
            logger.error(f"Error handling graph event: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error("Error processing response")
            # Ensure spinner is hidden even on error
            await self.send_json({
                'type': 'processing_complete'
            })

    async def handle_initial_conversation(self, user_input: str, conversation_history: list):
        """Handle initial conversation before order context"""
        try:
            # Check for order-related keywords
            order_mentioned = await self.check_for_order_mention(user_input)

            if order_mentioned:
                # Get user's recent orders
                recent_orders = await self.db_ops.get_recent_orders(self.user_id)

                if recent_orders:
                    # Ask user to specify which order they want to discuss
                    order_list = self.format_order_list(recent_orders)
                    await self.send_json({
                        'type': 'agent_response',
                        'message': f"I see you'd like to discuss an order. Here are your recent orders:\n\n{order_list}\n\nWhich order would you like to discuss? You can refer to the order by its number."
                    })
                    self.conversation_state = 'awaiting_order_selection'
                else:
                    await self.send_json({
                        'type': 'agent_response',
                        'message': "I notice you'd like to discuss an order, but I don't see any recent orders in your account. Could you please provide an order number, or let me know if you need help with something else?"
                    })
            else:
                # Handle general conversation
                response = await self.handle_general_conversation(user_input)
                await self.send_json({
                    'type': 'agent_response',
                    'message': response
                })

        except Exception as e:
            logger.error(f"Error in initial conversation: {str(e)}")
            await self.send_error("Error processing your message")

    async def check_for_order_mention(self, user_input: str) -> bool:
        """Check if user's message mentions order-related terms"""
        order_keywords = ['order', 'purchase', 'bought', 'delivery', 'tracking',
                          'shipped', 'package', 'return', 'cancel']
        return any(keyword in user_input.lower() for keyword in order_keywords)

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

    async def handle_general_conversation(self, user_input: str) -> str:
        """Handle general conversation before order context"""
        greetings = ['hi', 'hello', 'hey', 'good morning',
                     'good afternoon', 'good evening']

        if any(greeting in user_input.lower() for greeting in greetings):
            return ("Hello! I'm your order support assistant. I can help you with:\n"
                    "- Checking order status and tracking\n"
                    "- Modifying orders\n"
                    "- Cancellations and returns\n"
                    "- General order support\n\n"
                    "Do you have a specific order you'd like to discuss?")

        return ("I'm here to help with your orders! Would you like to:\n"
                "1. Check a recent order\n"
                "2. Track a shipment\n"
                "3. Modify an order\n"
                "4. Get help with something else\n\n"
                "Just let me know what you need!")

    async def handle_confirmation_request(self, message: AIMessage, conversation: Any):
        """Handle requests requiring user confirmation"""
        try:
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            if not tool_calls:
                return

            tool_call = tool_calls[0]
            args = json.loads(tool_call['function']['arguments'])

            # Format confirmation message
            confirmation_message = self.format_confirmation_message(
                tool_call['function']['name'],
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
            function_info = tool_call.get('function', {})
            tool_name = function_info.get('name')
            tool_args = json.loads(function_info.get('arguments', '{}'))

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

        tool_name = message.additional_kwargs['tool_calls'][0]['function']['name']
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

    async def send_json(self, data: Dict[str, Any]):
        """Send JSON response to WebSocket"""
        await self.send(text_data=json.dumps(data))

    async def send_error(self, message: str):
        """Send error message and hide spinner"""
        await self.send_json({
            'type': 'error',
            'message': message
        })
        await self.send_json({
            'type': 'processing_complete'
        })
