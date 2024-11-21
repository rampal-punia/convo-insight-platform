"""
Conversation context management for order support system.
Handles state tracking, context persistence, and conversation flow management.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging
import traceback
from django.core.cache import cache
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction

logger = logging.getLogger('orders')

User = get_user_model()

"""
Hybrid conversation context management combining cache and database storage.
"""


class CustomJSONEncoder(DjangoJSONEncoder):
    """Enhanced JSON encoder that handles User objects and datetime"""

    def default(self, obj):
        if isinstance(obj, User):
            return {
                'id': obj.id,
                'username': obj.username,
                'email': obj.email
            }
        return super().default(obj)


@dataclass
class ConversationStateData:
    """Rich conversation state data structure for in-memory/cache use"""
    conversation_id: str
    current_intent: Optional[str] = None
    previous_intent: Optional[str] = None
    active_order_id: Optional[str] = None  # Currently active order
    last_mentioned_order_id: Optional[str] = None  # Last mentioned order
    last_message_time: Optional[datetime] = None
    last_action: Optional[str] = None
    extracted_entities: Dict = field(default_factory=dict)
    pending_confirmations: list = field(default_factory=list)
    active_tools: list = field(default_factory=list)
    context_variables: Dict = field(default_factory=dict)
    order_context: Dict = field(default_factory=dict)
    conversation_metrics: Dict = field(default_factory=dict)
    last_snapshot_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert state to a JSON-serializable dictionary"""
        state_dict = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.last_message_time:
            state_dict['last_message_time'] = self.last_message_time.isoformat()
        if self.last_snapshot_time:
            state_dict['last_snapshot_time'] = self.last_snapshot_time.isoformat()
        return state_dict


class ConversationContextManager:
    """Manages conversation context with hybrid storage approach"""

    def __init__(self, db_ops):
        self.db_ops = db_ops
        self.cache_timeout = 3600  # 1 hour
        self.snapshot_interval = 300  # 5 minutes

    async def initialize_conversation(self, conversation_id: str, user_id: str) -> Tuple[bool, Optional[str]]:
        """Initialize or get conversation with proper error handling"""
        try:
            from convochat.models import Conversation

            @database_sync_to_async
            def get_or_create_conversation():
                with transaction.atomic():
                    conversation, created = Conversation.objects.get_or_create(
                        id=conversation_id,
                        defaults={
                            'user': self.db_ops.user,
                            'title': 'Order Support Conversation',
                            'status': 'AC'
                        }
                    )
                    return conversation, created

            conversation, created = await get_or_create_conversation()

            # Initialize state if created
            if created:
                initial_state = ConversationStateData(
                    conversation_id=conversation_id)
                await self._cache_state(conversation_id, initial_state)

            return True, conversation.id

        except Exception as e:
            logger.error(f"Error initializing conversation: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None

    async def _get_cached_state(self, conversation_id: str) -> Optional[ConversationStateData]:
        """Retrieve state from cache with proper error handling"""
        try:
            cache_key = f"conversation_state_{conversation_id}"
            cached_data = cache.get(cache_key)

            if cached_data:
                try:
                    data = json.loads(cached_data)
                    return self._deserialize_state(data)
                except json.JSONDecodeError:
                    logger.error("Failed to decode cached state JSON")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached state: {str(e)}")
            return None

    def _serialize_state(self, state: ConversationStateData) -> str:
        """Serialize state data to JSON string"""
        try:
            state_dict = state.to_dict()
            return json.dumps(state_dict, cls=CustomJSONEncoder)
        except Exception as e:
            logger.error(f"Error serializing state: {str(e)}")
            raise

    def _deserialize_state(self, data: Dict[str, Any]) -> ConversationStateData:
        """Deserialize state data with proper type checking"""
        try:
            # Convert ISO format strings back to datetime objects
            if 'last_message_time' in data and data['last_message_time']:
                data['last_message_time'] = datetime.fromisoformat(
                    data['last_message_time'])
            if 'last_snapshot_time' in data and data['last_snapshot_time']:
                data['last_snapshot_time'] = datetime.fromisoformat(
                    data['last_snapshot_time'])

            # Remove conversation_id from the data dict to avoid duplicate argument
            conversation_id = data.pop('conversation_id')

            return ConversationStateData(
                conversation_id=conversation_id,
                **data
            )
        except Exception as e:
            logger.error(f"Error deserializing state: {str(e)}")
            # Return a new state object if deserialization fails
            return ConversationStateData(conversation_id=data.get('conversation_id', 'unknown'))

    async def _cache_state(self, conversation_id: str, state: ConversationStateData):
        """Cache conversation state."""
        try:
            cache_key = f"conversation_state_{conversation_id}"
            serialized_state = self._serialize_state(state)
            cache.set(cache_key, serialized_state, self.cache_timeout)
        except Exception as e:
            logger.error(f"Error caching state: {str(e)}")
            logger.error(traceback.format_exc())

    @database_sync_to_async
    def _save_snapshot(self, conversation_id: str, state: ConversationStateData):
        try:
            from support_agent.models import ConversationSnapshot
            from convochat.models import Conversation

            with transaction.atomic():
                conversation = self.db_ops.get_or_create_conversation(
                    conversation_id,
                    state.order_context.get('order_id')
                )

                metrics = state.conversation_metrics or {}

                # Create snapshot with complete state
                snapshot = ConversationSnapshot.objects.acreate(
                    conversation=conversation,
                    state_data=state.to_dict(),
                    metrics_data=metrics,
                    total_messages=metrics.get('total_messages', 0),
                    user_messages=metrics.get('user_messages', 0),
                    ai_messages=metrics.get('ai_messages', 0),
                    intent_changes=metrics.get('intent_changes', 0),
                    tool_uses=metrics.get('tool_uses', 0),
                    current_intent=state.current_intent,
                    previous_intent=state.previous_intent
                )

                return snapshot

        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def update_context(
        self,
        conversation_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update context with improved order handling"""
        try:
            # Ensure conversation exists
            success, _ = await self.initialize_conversation(
                conversation_id,
                str(self.db_ops.user.id)
            )
            if not success:
                return False

            # Get current state
            state = await self._get_cached_state(conversation_id)
            if not state:
                state = ConversationStateData(conversation_id=conversation_id)

            # Handle order context updates
            if 'order_info' in updates:
                order_info = updates['order_info']
                if order_info:
                    state.active_order_id = order_info.get('order_id')
                    state.last_mentioned_order_id = order_info.get('order_id')
                    state.order_context = order_info

            # Handle intent changes
            if 'current_intent' in updates and updates['current_intent'] != state.current_intent:
                state.previous_intent = state.current_intent
                state.current_intent = updates['current_intent']
                if 'conversation_metrics' not in state.conversation_metrics:
                    state.conversation_metrics['intent_changes'] = 0
                state.conversation_metrics['intent_changes'] += 1

            # Update all other fields
            for key, value in updates.items():
                if key != 'order_info' and key != 'current_intent':
                    if hasattr(state, key):
                        setattr(state, key, value)
                    else:
                        state.context_variables[key] = value

            # Update last modification time
            state.last_message_time = datetime.now(timezone.utc)

            # Cache updated state
            await self._cache_state(conversation_id, state)

            # Create snapshot if needed
            if self._should_save_snapshot(state):
                await self._create_state_snapshot(
                    conversation_id=conversation_id,
                    state=state,
                    snapshot_type='AU'
                )
                state.last_snapshot_time = datetime.now(timezone.utc)
                await self._cache_state(conversation_id, state)

            return True

        except Exception as e:
            logger.error(f"Error updating context: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _should_save_snapshot(self, state: ConversationStateData) -> bool:
        """Determine if a new snapshot should be saved"""
        if not state.last_snapshot_time:
            return True

        time_since_last = (datetime.now(timezone.utc) -
                           state.last_snapshot_time).total_seconds()
        return time_since_last >= self.snapshot_interval

    async def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get current context with improved error handling"""
        try:
            # Ensure conversation exists
            success, _ = await self.initialize_conversation(conversation_id, str(self.db_ops.user.id))
            if not success:
                return {'conversation_id': conversation_id}

            # Try cache first
            state = await self._get_cached_state(conversation_id)
            if state:
                return state.to_dict()

            # Create new state if none exists
            state = ConversationStateData(conversation_id=conversation_id)
            await self._cache_state(conversation_id, state)
            return state.to_dict()

        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            logger.error(traceback.format_exc())
            return {'conversation_id': conversation_id}

    @database_sync_to_async
    def _ensure_conversation_exists(self, conversation_id: str) -> bool:
        """Ensure conversation exists in database"""
        try:
            from convochat.models import Conversation
            conversation, created = Conversation.objects.get_or_create(
                id=conversation_id,
                defaults={
                    'user': self.db_ops.user,
                    'title': 'Order Support Conversation',
                    'status': 'AC'
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error ensuring conversation exists: {str(e)}")
            return False

    async def _create_state_snapshot(self, conversation_id: str, state: ConversationStateData, snapshot_type: str = 'AU'):
        """Create a detailed point-in-time snapshot of the conversation state"""
        try:
            # First ensure conversation exists
            exists = await self._ensure_conversation_exists(conversation_id)
            if not exists:
                logger.error(
                    "Cannot create snapshot: conversation does not exist")
                return None

            # Save snapshot to database
            snapshot = await self._save_snapshot(conversation_id, state)
            if snapshot:
                # Update last snapshot time in state
                state.last_snapshot_time = datetime.now(timezone.utc)
                await self._cache_state(conversation_id, state)

            return snapshot

        except Exception as e:
            logger.error(f"Error creating state snapshot: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def save_conversation_state(self, conversation_id: str, final_state: Dict[str, Any]) -> bool:
        """Save final conversation state and cleanup"""
        try:
            # Ensure conversation exists
            exists = await self._ensure_conversation_exists(conversation_id)
            if not exists:
                logger.error("Cannot save state: conversation does not exist")
                return False

            # Get current state
            state = await self._get_cached_state(conversation_id)
            if not state:
                state = ConversationStateData(conversation_id=conversation_id)

            # Update state with final data
            for key, value in final_state.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    state.context_variables[key] = value

            # Create final snapshot
            snapshot = await self._create_state_snapshot(
                conversation_id=conversation_id,
                state=state,
                snapshot_type='FN'  # Final snapshot
            )

            if snapshot:
                # Update conversation record
                success = await self.db_ops.update_conversation(conversation_id, {
                    'status': 'EN',  # Ended
                    'summary': state.to_dict(),
                    'overall_sentiment_score': state.conversation_metrics.get('sentiment_score'),
                    'resolution_status': state.conversation_metrics.get('resolution_status', 'UN')
                })

                if success:
                    # Clear cache
                    cache_key = f"conversation_state_{conversation_id}"
                    cache.delete(cache_key)
                    return True

            return False

        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
            logger.error(traceback.format_exc())
            return False
