"""
Intent detection and routing for order support conversations.
Implements hybrid approach using rules and LLM for accurate intent classification.
"""

import logging
import traceback
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger('orders')


@dataclass
class IntentScore:
    """Represents a scored intent with confidence."""
    intent: str
    confidence: float
    sub_intents: List[str] = None
    entities: Dict[str, Any] = None


class IntentRouter:
    """
    Handles intent detection and routing for order support conversations.
    Uses a hybrid approach combining rule-based patterns and LLM for intent classification.
    """

    # Define core intents with their patterns and variations
    INTENT_PATTERNS = {
        'track_order': [
            'where is my order',
            'track order',
            'track package',
            'shipping status',
            'delivery status',
            'when will it arrive',
            'package location',
            'delivery time',
            'shipping update',
            'order location',
            'track shipment',
            'check delivery',
            'delivery progress',
            'order tracking',
            'shipping information'
        ],
        'modify_order_quantity': [
            'change quantity',
            'modify order',
            'update quantity',
            'change amount',
            'order quantity',
            'increase quantity',
            'decrease quantity',
            'reduce amount',
            'add more items',
            'remove items',
            'adjust quantity',
            'change number of items',
            'modify item count',
            'update item quantity'
        ],
        'cancel_order': [
            'cancel order',
            'cancel my order',
            'stop order',
            'don\'t want order',
            'withdraw order',
            'terminate order'
        ],
        'delivery_issue': [
            'not received',
            'wrong address',
            'delivery problem',
            'missing package',
            'late delivery',
            'delivery delay',
            'address wrong',
            'wrong delivery'
        ],
        'order_detail': [
            'order details',
            'what did i order',
            'order information',
            'show order',
            'order summary',
            'purchase details',
            'items ordered'
        ]
    }

    # Intent relationship mapping for handling related intents
    INTENT_RELATIONSHIPS = {
        'track_order': ['delivery_issue', 'order_detail'],
        'modify_order_quantity': ['order_detail', 'cancel_order'],
        'cancel_order': ['modify_order_quantity', 'order_detail'],
        'delivery_issue': ['track_order', 'order_detail'],
        'order_detail': ['track_order', 'modify_order_quantity']
    }

    def __init__(self, llm=None):
        """Initialize intent router with optional custom LLM."""
        self.llm = llm
        self.default_intent = 'order_detail'
        self._setup_intent_detection_prompt()

    def _setup_intent_detection_prompt(self):
        """Set up the prompt template for LLM-based intent detection."""
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in detecting customer service intents for an e-commerce order support system.
            Available intents are: track_order, modify_order_quantity, cancel_order, delivery_issue, and order_detail.
            
            Analyze the user message and determine the most likely intent(s). Consider:
            1. Primary intent: The main purpose of the user's message
            2. Secondary intents: Related or implied intents
            3. Any specific entities mentioned (order IDs, quantities, dates, etc.)
            
            Previous conversation context:
            {conversation_history}
            
            Current order context:
            {order_context}
            
            Respond in the following JSON format:
            {
                "primary_intent": "intent_name",
                "confidence": 0.0-1.0,
                "secondary_intents": ["intent1", "intent2"],
                "entities": {
                    "order_id": "value",
                    "quantity": "value",
                    "date": "value"
                }
            }"""),
            ("human", "{user_input}")
        ])

    async def detect_intent(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> IntentScore:
        """
        Detect intent from user input using multiple detection methods.

        Args:
            user_input: The user's message
            context: Current conversation context including history

        Returns:
            IntentScore object containing detected intent and confidence
        """
        try:
            # First try rule-based pattern matching
            rule_based_result = self._rule_based_detection(user_input.lower())
            if rule_based_result and rule_based_result.confidence > 0.8:
                # If we detect order_detail intent, ensure we have order context
                if rule_based_result.intent == 'order_detail':
                    # Use _extract_entities to get order ID
                    extracted_entities = self._extract_entities(
                        user_input, 'order_detail', context)
                    order_id = context.get(
                        'order_id') or extracted_entities.get('order_id')
                    if order_id:
                        rule_based_result.entities = {'order_id': order_id}
                return rule_based_result

            # Use LLM for more nuanced intent detection
            llm_result = await self._llm_based_detection(user_input, context)
            if llm_result:
                # Combine results if we have both
                if rule_based_result:
                    return self._combine_intent_results(rule_based_result, llm_result)
                return llm_result

            # Fall back to rule-based result or default
            if rule_based_result:
                return rule_based_result

            # If we're in an order context, default to order_detail
            if context.get('order_id'):
                return IntentScore(
                    intent='order_detail',
                    confidence=0.6,
                    entities={'order_id': context['order_id']}
                )

            return IntentScore(
                intent=self.default_intent,
                confidence=0.5
            )

        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            logger.error(traceback.format_exc())
            return IntentScore(
                intent=self.default_intent,
                confidence=0.5,
                # Still try to extract entities
                entities=self._extract_entities(
                    user_input, self.default_intent, context)
            )

    def _rule_based_detection(self, user_input: str) -> Optional[IntentScore]:
        """
        Detect intent using rule-based pattern matching with confidence scoring.

        Args:
            user_input: Lowercase user message

        Returns:
            IntentScore object or None
        """
        scores = []
        for intent, patterns in self.INTENT_PATTERNS.items():
            confidence = self._calculate_pattern_confidence(
                user_input, patterns)
            if confidence > 0:
                scores.append((intent, confidence))

        if not scores:
            return None

        # Sort by confidence and get best match
        scores.sort(key=lambda x: x[1], reverse=True)
        best_intent, confidence = scores[0]

        # Get related intents as sub-intents
        sub_intents = [
            intent for intent, _ in scores[1:3]
            if intent in self.INTENT_RELATIONSHIPS.get(best_intent, [])
        ]

        # Extract relevant entities
        entities = self._extract_entities(user_input, best_intent)

        return IntentScore(
            intent=best_intent,
            confidence=confidence,
            sub_intents=sub_intents,
            entities=entities
        )

    def _calculate_pattern_confidence(
        self,
        user_input: str,
        patterns: List[str]
    ) -> float:
        """
        Calculate confidence score for pattern matching.

        Args:
            user_input: User's message
            patterns: List of patterns to match against

        Returns:
            Confidence score between 0 and 1
        """
        max_score = 0
        for pattern in patterns:
            # Calculate Levenshtein distance for fuzzy matching
            pattern_words = set(pattern.split())
            input_words = set(user_input.split())

            # Calculate word overlap
            overlap = len(pattern_words.intersection(input_words))
            total_words = len(pattern_words.union(input_words))

            # Calculate score based on overlap and exact matches
            exact_match = pattern in user_input
            score = (overlap / total_words) * 0.8
            if exact_match:
                score += 0.2

            max_score = max(max_score, score)

        return max_score

    async def _llm_based_detection(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> Optional[IntentScore]:
        """
        Use LLM for intent detection with context awareness.

        Args:
            user_input: User's message
            context: Conversation context

        Returns:
            IntentScore object or None
        """
        try:
            if not self.llm:
                return None

            # Format conversation history
            conversation_history = self._format_conversation_history(
                context.get('history', [])
            )

            # Get LLM response
            response = await self.llm.ainvoke(
                self.intent_prompt.format(
                    user_input=user_input,
                    conversation_history=conversation_history,
                    order_context=context.get('order_info', {})
                )
            )

            # Parse LLM response
            result = self._parse_llm_response(response.content)
            if not result:
                return None

            return IntentScore(
                intent=result['primary_intent'],
                confidence=result['confidence'],
                sub_intents=result.get('secondary_intents', []),
                entities=result.get('entities', {})
            )

        except Exception as e:
            logger.error(f"Error in LLM-based detection: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse and validate LLM response."""
        try:
            import json
            # Extract JSON from response if needed
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if not json_match:
                return None

            result = json.loads(json_match.group())

            # Validate required fields
            if 'primary_intent' not in result or 'confidence' not in result:
                return None

            # Validate intent
            if result['primary_intent'] not in self.INTENT_PATTERNS:
                return None

            return result

        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response JSON")
            logger.error(traceback.format_exc())
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _combine_intent_results(
        self,
        rule_result: IntentScore,
        llm_result: IntentScore
    ) -> IntentScore:
        """
        Combine rule-based and LLM-based results.

        Args:
            rule_result: Result from rule-based detection
            llm_result: Result from LLM-based detection

        Returns:
            Combined IntentScore
        """
        # If intents match, combine confidence and entities
        if rule_result.intent == llm_result.intent:
            return IntentScore(
                intent=rule_result.intent,
                confidence=(rule_result.confidence +
                            llm_result.confidence) / 2,
                sub_intents=list(set(rule_result.sub_intents or [] +
                                     llm_result.sub_intents or [])),
                entities={**(rule_result.entities or {}),
                          **(llm_result.entities or {})}
            )

        # If different intents, use the one with higher confidence
        return rule_result if rule_result.confidence > llm_result.confidence else llm_result

    def _extract_entities(self, user_input: str, intent: str, context: Dict = None) -> Dict[str, Any]:
        """Extract and consolidate all relevant entities with context awareness"""
        entities = {}
        context = context or {}

        # First check if we have entities in context
        stored_entities = context.get('extracted_entities', {})

        # Extract order IDs with comprehensive patterns
        order_patterns = [
            r'order\s*#?\s*(\d+)',
            r'#\s*(\d+)',
            r'number\s*(\d+)',
            r'^(\d+)$',
            r'order\s+id\s*:\s*(\d+)',
            r'orderid\s*:\s*(\d+)',
            r'order\s+(\d+)',
        ]

        # Try to extract order ID or use from context
        order_id = stored_entities.get('order_id')
        if not order_id:
            for pattern in order_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    order_id = match.group(1)
                    break

        if order_id:
            entities['order_id'] = order_id

        # Extract quantities for modification intents
        if intent == 'modify_order_quantity':
            quantity = stored_entities.get('quantity')
            if not quantity:
                quantity_matches = re.findall(
                    r'(\d+)\s*(?:items?|pieces?|qty)',
                    user_input.lower()
                )
                if quantity_matches:
                    quantity = int(quantity_matches[0])
            if quantity:
                entities['quantity'] = quantity

        # Extract dates or use from context
        date = stored_entities.get('date')
        if not date:
            date_patterns = [
                r'(\d{1,2}/\d{1,2}/\d{2,4})',  # MM/DD/YYYY
                r'(\d{4}-\d{1,2}-\d{1,2})',     # YYYY-MM-DD
            ]
            for pattern in date_patterns:
                match = re.search(pattern, user_input)
                if match:
                    date = match.group(1)
                    break
        if date:
            entities['date'] = date

        # Merge with existing entities from context while prioritizing new ones
        if stored_entities:
            for key, value in stored_entities.items():
                if key not in entities:  # Don't overwrite new entities
                    entities[key] = value

        return entities

    def _format_conversation_history(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """Format conversation history for LLM context."""
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-3:]:  # Only use last 3 messages for context
            role = "User" if msg.get('is_user') else "Assistant"
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    async def validate_intent_transition(
        self,
        current_intent: str,
        new_intent: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Validate if transitioning between intents is logical.

        Args:
            current_intent: Current conversation intent
            new_intent: Detected new intent
            context: Conversation context

        Returns:
            Boolean indicating if transition is valid
        """
        # Always allow transition to related intents
        if new_intent in self.INTENT_RELATIONSHIPS.get(current_intent, []):
            return True

        # Allow transition if confidence is high enough
        confidence_threshold = 0.8
        result = await self.detect_intent(context.get('last_message', ''), context)
        return result.confidence >= confidence_threshold
