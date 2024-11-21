from typing import Dict, Any, Optional, Tuple
import re
import logging
from datetime import datetime, timezone
import traceback
from dataclasses import dataclass

logger = logging.getLogger('orders')


@dataclass
class ConversationState:
    state: str
    intent: Optional[str] = None
    order_id: Optional[str] = None
    order_info: Optional[Dict] = None
    context: Dict[str, Any] = None


class ConversationFlowManager:
    """Manages conversation flow and state transitions"""

    def __init__(self, user_name, user_id, conversation_id, db_ops, context_manager, intent_router):
        self.user_id = user_id
        self.user_name = user_name
        self.conversation_id = conversation_id
        self.db_ops = db_ops
        self.context_manager = context_manager
        self.intent_router = intent_router

    async def process_message(self, user_input: str, current_state: str) -> Tuple[ConversationState, Optional[str]]:
        """
        Process user message based on current conversation state
        Returns: (new_state, response_message)
        """
        try:
            # Check for explicit order mentions first
            mentioned_order_id = self._extract_order_id(user_input)

            # Get current context
            context = await self.context_manager.get_context(self.conversation_id)
            current_order_id = context.get('order_id')

            # If no explicit order mentioned, use the last active order
            effective_order_id = mentioned_order_id or current_order_id

            # Detect intent
            intent_result = await self.intent_router.detect_intent(user_input, context)
            logger.info(
                f"Detected intent: {intent_result.intent} with confidence: {intent_result.confidence}")

            # If modifying/tracking/canceling, ensure we're using the correct order
            if intent_result.intent in ['modify_order_quantity', 'track_order', 'cancel_order']:
                if not effective_order_id:
                    return ConversationState(
                        state='awaiting_order_selection'
                    ), "Which order would you like to modify?"

                # Get order details
                order_info = await self.db_ops.get_order_details(effective_order_id)
                if not order_info:
                    return ConversationState(
                        state='error'
                    ), f"Could not find order #{effective_order_id}"

                # Update context with current order
                await self.context_manager.update_context(
                    self.conversation_id,
                    {
                        'order_id': effective_order_id,
                        'order_info': order_info,
                        'last_mentioned_order_id': effective_order_id
                    }
                )

                return ConversationState(
                    state='active',
                    intent=intent_result.intent,
                    order_id=effective_order_id,
                    order_info=order_info,
                    context={'last_action': 'order_reference'}
                ), None

            # Process based on current state
            if current_state == 'initial':
                return await self._handle_initial_state(user_input)
            elif current_state == 'awaiting_order_selection':
                return await self._handle_order_selection(user_input)
            elif current_state == 'active':
                return await self._handle_active_state(
                    user_input, effective_order_id, intent_result
                )

            logger.error(f"Unknown conversation state: {current_state}")
            return ConversationState(
                state='initial'
            ), "I apologize, but I'm having trouble with your request. Could you please start over?"

        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            logger.error(traceback.format_exc())
            return ConversationState(
                state='initial'
            ), "I encountered an error. Could you please try again?"

    async def _handle_initial_state(self, user_input: str) -> Tuple[ConversationState, str]:
        """Handle messages in initial state"""
        # Check for order mentions
        order_id = self._extract_order_id(user_input)
        if order_id:
            return await self._process_order_reference(order_id)

        # Check for order-related keywords
        if self._contains_order_keywords(user_input):
            recent_orders = await self.db_ops.get_recent_orders(self.user_id)
            if recent_orders:
                order_list = self._format_order_list(recent_orders)
                return (
                    ConversationState(state='awaiting_order_selection'),
                    f"I see you'd like to discuss an order. Here are your recent orders:\n\n{order_list}\n\nWhich order would you like to discuss?"
                )

        # Default to general conversation
        return (
            ConversationState(state='initial'),
            f"Hello {self.user_name}! I can help you with your orders. Do you have a specific order number you'd like to discuss?"
        )

    async def _handle_order_selection(self, user_input: str) -> Tuple[ConversationState, str]:
        """Handle order selection responses"""
        order_id = self._extract_order_id(user_input)
        if order_id:
            return await self._process_order_reference(order_id)

        return (
            ConversationState(state='awaiting_order_selection'),
            "I couldn't identify an order number. Could you please provide the order number you'd like to discuss?"
        )

    async def _handle_active_state(
        self,
        user_input: str,
        order_id: Optional[str],
        intent_result: Any
    ) -> Tuple[ConversationState, Optional[str]]:
        """Handle messages when conversation is active with proper intent and entity tracking"""
        try:
            # Get current context
            context = await self.context_manager.get_context(self.conversation_id)

            # Detect intent from user input
            intent_result = await self.intent_router.detect_intent(user_input, context)
            logger.info(
                f"Detected intent: {intent_result.intent} with confidence: {intent_result.confidence}")

            # Extract and consolidate entities with context awareness
            entities = intent_result.entities or {}

            # Determine the correct order_id with priority:
            # 1. Explicitly mentioned in current message
            # 2. Active order from context
            # 3. Last mentioned order
            current_order_id = (
                # Check current message
                self._extract_order_id(user_input) or
                # Then check active order
                context.get('active_order_id') or
                # Then passed order_id
                order_id or
                # Finally last mentioned
                context.get('last_mentioned_order_id')
            )

            # Get order information if we have an order ID
            order_info = None
            if current_order_id:
                order_info = await self.db_ops.get_order_details(current_order_id)
                if order_info:
                    # Update context with current order information
                    await self.context_manager.update_context(
                        self.conversation_id,
                        {
                            'active_order_id': current_order_id,
                            'last_mentioned_order_id': current_order_id,
                            'order_info': order_info
                        }
                    )

            # If we need order context but don't have it, request it
            if self._intent_requires_order(intent_result.intent) and not order_info:
                recent_orders = await self.db_ops.get_recent_orders(self.user_id)
                if recent_orders:
                    order_list = self._format_order_list(recent_orders)
                    return ConversationState(
                        state='awaiting_order_selection',
                        intent=intent_result.intent
                    ), f"Which order would you like to modify?\n\n{order_list}"
                return ConversationState(
                    state='awaiting_order_selection',
                    intent=intent_result.intent
                ), "Please provide the order number you'd like to modify."

            # Update context with new information
            context_updates = {
                'current_intent': intent_result.intent,
                'previous_intent': context.get('current_intent'),
                'extracted_entities': entities,
                'last_message_time': datetime.now(timezone.utc)
            }

            if order_info:
                context_updates['order_info'] = order_info

            await self.context_manager.update_context(
                self.conversation_id,
                context_updates
            )

            return ConversationState(
                state='active',
                intent=intent_result.intent,
                order_id=current_order_id,
                order_info=order_info,
                context={**context, **context_updates}
            ), None

        except Exception as e:
            logger.error(f"Error in _handle_active_state: {str(e)}")
            logger.error(traceback.format_exc())
            return ConversationState(state='error'), "I encountered an error. Could you please try again?"

    async def _process_order_reference(self, order_id: str) -> Tuple[ConversationState, str]:
        """Process a reference to a specific order and update context"""
        try:
            # Validate order exists and get details
            order_info = await self.db_ops.get_order_details(order_id)
            if not order_info:
                return (
                    ConversationState(state='initial'),
                    f"I couldn't find order #{order_id}. Could you please verify the order number?"
                )

            # Get current context
            context = await self.context_manager.get_context(self.conversation_id)

            # Set up base context updates
            updates = {
                'order_id': order_id,
                'order_info': order_info,
                'last_mentioned_order_id': order_id,
                'last_message_time': datetime.now(timezone.utc)
            }

            # Update conversation context
            await self.context_manager.update_context(
                self.conversation_id,
                updates
            )

            # Initialize with neutral state until we get user's specific intent
            new_state = ConversationState(
                state='active',
                order_id=order_id,
                order_info=order_info,
                context=updates
            )

            return new_state, (
                f"I found your order #{order_id}. "
                "I can help you with tracking this order, checking its details, "
                "modifying quantities, or handling cancellations. What would you like to know?"
            )

        except Exception as e:
            logger.error(f"Error processing order reference: {str(e)}")
            logger.error(traceback.format_exc())
            return ConversationState(state='error'), "I encountered an error. Could you please try again?"

    def _extract_order_id(self, text: str) -> Optional[str]:
        """Extract order ID from text"""
        patterns = [
            r'order\s*#?\s*(\d+)',
            r'#\s*(\d+)',
            r'number\s*(\d+)',
            r'^(\d+)$'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _contains_order_keywords(self, text: str) -> bool:
        """Check if text contains order-related keywords"""
        keywords = [
            'order', 'purchase', 'bought', 'delivery',
            'tracking', 'shipped', 'package', 'return', 'cancel'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    def _format_order_list(self, orders: list) -> str:
        """Format list of orders for display"""
        formatted = []
        for order in orders:
            formatted.append(
                f"Order #{order['id']} - Placed on {order['created_date']}\n"
                f"Status: {order['status']}\n"
                f"Items: {order['item_count']}\n"
                f"Total: ${order['total_amount']}"
            )
        return "\n\n".join(formatted)

    async def _find_last_discussed_order(self) -> Optional[str]:
        """Find the last order discussed in the conversation history"""
        try:
            history = await self.db_ops.get_conversation_history(self.conversation_id)
            for message in reversed(history):
                order_id = self._extract_order_id(message.content)
                if order_id:
                    return order_id
            return None
        except Exception as e:
            logger.error(f"Error finding last order: {str(e)}")
            return None

    def _intent_requires_order(self, intent: str) -> bool:
        """Determine if an intent requires order context"""
        order_required_intents = {
            'track_order',
            'modify_order_quantity',
            'cancel_order',
            'order_detail',
            'delivery_issue'
        }
        return intent in order_required_intents
