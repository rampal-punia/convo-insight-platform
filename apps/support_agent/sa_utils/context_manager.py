"""
Conversation context management for order support system.
Handles state tracking, context persistence, and conversation flow management.
"""

import json
import logging
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from django.core.cache import cache
from channels.db import database_sync_to_async

from ..models import ConversationSnapshot

logger = logging.getLogger('orders')

"""
Hybrid conversation context management combining cache and database storage.
"""


@dataclass
class ConversationStateData:
    """Rich conversation state data structure for in-memory/cache use"""
    conversation_id: str
    current_intent: str = None
    previous_intent: str = None
    last_message_time: datetime = None
    last_action: str = None
    extracted_entities: dict = None
    pending_confirmations: list = None
    active_tools: list = None
    context_variables: dict = None
    order_context: dict = None
    user_preferences: dict = None
    conversation_metrics: dict = None
    last_snapshot_time: datetime = None

    def __post_init__(self):
        self.extracted_entities = self.extracted_entities or {}
        self.pending_confirmations = self.pending_confirmations or []
        self.active_tools = self.active_tools or []
        self.context_variables = self.context_variables or {}
        self.order_context = self.order_context or {}
        self.user_preferences = self.user_preferences or {}
        self.conversation_metrics = self.conversation_metrics or {}


class ConversationContextManager:
    def __init__(self, db_ops):
        self.db_ops = db_ops
        self.cache_timeout = 3600  # 1 hour
        self.snapshot_interval = 300  # 5 minutes

    async def _get_cached_state(self, conversation_id: str) -> Optional[ConversationStateData]:
        """Retrieve state from cache."""
        cache_key = f"conversation_state_{conversation_id}"
        cached_data = cache.get(cache_key)

        if cached_data:
            # Convert cached JSON data back to ConversationStateData
            data = json.loads(cached_data)
            # Convert string timestamps back to datetime objects
            if data.get('last_message_time'):
                data['last_message_time'] = datetime.fromisoformat(
                    data['last_message_time'])
            if data.get('last_snapshot_time'):
                data['last_snapshot_time'] = datetime.fromisoformat(
                    data['last_snapshot_time'])
            return ConversationStateData(**data)
        return None

    async def _cache_state(self, conversation_id: str, state: ConversationStateData):
        """Cache conversation state."""
        cache_key = f"conversation_state_{conversation_id}"
        # Convert datetime objects to ISO format strings for JSON serialization
        state_dict = asdict(state)
        if state.last_message_time:
            state_dict['last_message_time'] = state.last_message_time.isoformat()
        if state.last_snapshot_time:
            state_dict['last_snapshot_time'] = state.last_snapshot_time.isoformat()

        cache.set(cache_key, json.dumps(state_dict), self.cache_timeout)

    @database_sync_to_async
    def _load_persistent_state(self, conversation_id: str) -> Optional[ConversationStateData]:
        """Load minimal persistent state from database."""
        try:
            db_state = ConversationSnapshot.objects.get(
                conversation__id=conversation_id)
            # Create a full state object with persisted data
            return ConversationStateData(
                conversation_id=conversation_id,
                current_intent=db_state.current_intent,
                conversation_metrics=db_state.conversation_metrics,
                extracted_entities=db_state.critical_entities
            )
        except ConversationSnapshot.DoesNotExist:
            return None

    @database_sync_to_async
    def _save_persistent_state(self, conversation_id: str, state: ConversationStateData):
        """Save only critical state data to database."""
        # Only save essential data that needs to persist
        ConversationSnapshot.objects.update_or_create(
            conversation__id=conversation_id,
            defaults={
                'current_intent': state.current_intent,
                'conversation_metrics': {
                    'total_messages': state.conversation_metrics.get('total_messages', 0),
                    'intent_changes': state.conversation_metrics.get('intent_changes', 0),
                    'completion_status': state.conversation_metrics.get('completion_status')
                },
                'critical_entities': {
                    k: v for k, v in state.extracted_entities.items()
                    # Only persist critical entities
                    if k in {'order_id', 'product_id', 'customer_id'}
                }
            }
        )
        return None

    async def update_context(self, conversation_id: str, updates: Dict[str, Any]) -> bool:
        """Update context with cache priority."""
        try:
            # Get current state from cache first
            state = await self._get_cached_state(conversation_id)
            if not state:
                # If not in cache, try to load from database
                state = await self._load_persistent_state(conversation_id)
                if not state:
                    state = ConversationStateData(
                        conversation_id=conversation_id)

            # Update state with new data
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)

            # Always update cache
            await self._cache_state(conversation_id, state)

            # Only persist to database if critical data changed
            if self._has_critical_updates(updates):
                await self._save_persistent_state(conversation_id, state)

            return True

        except Exception as e:
            logger.error(f"Error updating context: {str(e)}")
            return False

    def _has_critical_updates(self, updates: Dict[str, Any]) -> bool:
        """Check if updates contain critical data that needs to persist."""
        critical_fields = {
            'current_intent',
            'conversation_metrics',
            'extracted_entities'
        }
        return bool(critical_fields.intersection(updates.keys()))

    async def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get current context with cache priority."""
        try:
            # Try cache first
            state = await self._get_cached_state(conversation_id)
            if state:
                return asdict(state)

            # Fall back to database
            state = await self._load_persistent_state(conversation_id)
            if state:
                await self._cache_state(conversation_id, state)
                return asdict(state)

            return {}

        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return {}

    @database_sync_to_async
    def _create_state_snapshot(self, conversation_id: str, state: ConversationStateData, snapshot_type: str = 'AU'):
        """Create a point-in-time snapshot of the state."""
        try:
            # Create snapshot using the class method
            snapshot = ConversationSnapshot.create_snapshot(
                conversation_id=conversation_id,
                state_data=asdict(state),
                metrics_data=state.conversation_metrics,
                snapshot_type=snapshot_type
            )

            # Update last snapshot time in state
            state.last_snapshot_time = datetime.now(timezone.utc)
            self._cache_state(conversation_id, state)

            return snapshot
        except Exception as e:
            logger.error(f"Error creating state snapshot: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def save_conversation_state(self, conversation_id: str, final_state: Dict[str, Any]) -> bool:
        """Save final state and cleanup."""
        try:
            state = await self._get_cached_state(conversation_id)
            if not state:
                return False

            # Create final snapshot
            await self._create_state_snapshot(
                conversation_id=conversation_id,
                state=state,
                snapshot_type='FN'  # Final snapshot
            )

            # Update conversation record
            await self.db_ops.update_conversation(conversation_id, {
                'status': 'EN',
                'summary': json.dumps(asdict(state)),
                'overall_sentiment_score': state.conversation_metrics.get('sentiment_score'),
                'resolution_status': state.conversation_metrics.get('resolution_status', 'UN')
            })

            # Clear cache
            cache_key = f"conversation_state_{conversation_id}"
            cache.delete(cache_key)

            return True

        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
            logger.error(traceback.format_exc())
            return False
