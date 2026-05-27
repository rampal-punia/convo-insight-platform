from channels.db import database_sync_to_async
import traceback
from django.utils import timezone
from typing import Dict, List, Optional, Tuple, Any
from django.db import transaction
from convochat.models import Conversation, Message, UserText, AIText
from langchain_core.messages import AIMessage, HumanMessage

from ..models import (
    Order,
    OrderConversationLink,
    OrderItem,
    OrderTracking
)
import logging
logger = logging.getLogger('orders')  # Get the orders logger


class DatabaseOperations:
    def __init__(self, user):
        self.user = user

    @database_sync_to_async
    def get_tracking_info(self, order_id: str) -> Dict:
        """Get comprehensive tracking information for an order"""
        try:
            order = Order.objects.select_related('user').get(
                id=order_id,
                user=self.user
            )

            # Get latest tracking update
            latest_tracking = order.tracking_history.order_by(
                '-timestamp').first()

            # Calculate estimated delivery window
            if order.estimated_delivery:
                delivery_window = self._calculate_delivery_window(
                    order.shipping_method,
                    order.estimated_delivery
                )
            else:
                delivery_window = None

            return {
                'order_id': str(order.id),
                'status': order.get_status_display(),
                'tracking_number': order.tracking_number,
                'carrier': order.carrier,
                'shipping_method': order.get_shipping_method_display(),
                'current_location': latest_tracking.location if latest_tracking else None,
                'estimated_delivery': order.estimated_delivery,
                'delivery_window': delivery_window,
                'shipped_date': order.shipped_date,
                'latest_update': {
                    'status': latest_tracking.get_status_display() if latest_tracking else None,
                    'timestamp': latest_tracking.timestamp if latest_tracking else None,
                    'description': latest_tracking.description if latest_tracking else None
                }
            }
        except Order.DoesNotExist:
            logger.warning(
                f"Order {order_id} not found for user {self.user.id}")
            logger.error(traceback.format_exc())
            return {"error": "Order not found"}
        except Exception as e:
            logger.error(f"Error getting tracking info: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error retrieving tracking information: {str(e)}"}

    @database_sync_to_async
    def update_conversation(self, conversation_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update conversation with the provided data

        Args:
            conversation_id: The ID of the conversation to update
            update_data: Dictionary containing fields to update

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            from convochat.models import Conversation

            conversation = Conversation.objects.get(id=conversation_id)

            # Update allowed fields
            if 'status' in update_data:
                conversation.status = update_data['status']

            if 'summary' in update_data:
                conversation.summary = update_data['summary']

            if 'overall_sentiment_score' in update_data:
                conversation.overall_sentiment_score = update_data['overall_sentiment_score']

            if 'resolution_status' in update_data:
                conversation.resolution_status = update_data['resolution_status']

            conversation.save()
            return True

        except Conversation.DoesNotExist:
            logger.error(f"Conversation {conversation_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    @database_sync_to_async
    def get_tracking_history(self, order_id: str) -> List[Dict]:
        """Get complete tracking history for an order"""
        try:
            tracking_events = OrderTracking.objects.filter(
                order_id=order_id,
                order__user=self.user
            ).order_by('-timestamp')

            return [
                {
                    'timestamp': event.timestamp,
                    'status': event.get_status_display(),
                    'location': event.location,
                    'description': event.description
                }
                for event in tracking_events
            ]
        except Exception as e:
            logger.error(f"Error getting tracking history: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    @database_sync_to_async
    def get_current_location(self, order_id: str) -> Dict:
        """Get current shipment location and status"""
        try:
            order = Order.objects.get(id=order_id, user=self.user)
            latest_tracking = order.tracking_history.order_by(
                '-timestamp').first()

            if not latest_tracking:
                return {
                    'location': None,
                    'status': order.get_status_display(),
                    'timestamp': None,
                }

            return {
                'location': latest_tracking.location,
                'status': latest_tracking.get_status_display(),
                'timestamp': latest_tracking.timestamp,
                'description': latest_tracking.description
            }
        except Order.DoesNotExist:
            logger.warning(
                f"Order {order_id} not found for user {self.user.id}")
            logger.error(traceback.format_exc())
            return {"error": "Order not found"}
        except Exception as e:
            logger.error(f"Error getting current location: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error retrieving location: {str(e)}"}

    @database_sync_to_async
    def get_delivery_estimate(self, order_id: str) -> Dict:
        """Get estimated delivery date and time window"""
        try:
            order = Order.objects.get(id=order_id, user=self.user)

            if not order.estimated_delivery:
                return {"error": "No delivery estimate available"}

            delivery_window = self._calculate_delivery_window(
                order.shipping_method,
                order.estimated_delivery
            )

            # Calculate confidence based on current status and tracking history
            confidence = self._calculate_delivery_confidence(order)

            return {
                'date': order.estimated_delivery,
                'time_window': delivery_window,
                'confidence': confidence,
                'shipping_method': order.get_shipping_method_display()
            }
        except Order.DoesNotExist:
            return {"error": "Order not found"}
        except Exception as e:
            logger.error(f"Error getting delivery estimate: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error retrieving delivery estimate: {str(e)}"}

    @database_sync_to_async
    def update_tracking_status(
        self,
        order_id: str,
        status: str,
        location: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict:
        """Update tracking status and add to tracking history"""
        try:
            with transaction.atomic():
                order = Order.objects.select_for_update().get(
                    id=order_id,
                    user=self.user
                )

                # Create new tracking event
                tracking_event = OrderTracking.objects.create(
                    order=order,
                    status=status,
                    location=location,
                    description=description
                )

                # Update order status if needed
                self._update_order_status_from_tracking(order, status)

                return {
                    'status': 'success',
                    'tracking_id': tracking_event.id,
                    'timestamp': tracking_event.timestamp
                }
        except Order.DoesNotExist:
            return {"error": "Order not found"}
        except Exception as e:
            logger.error(f"Error updating tracking status: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error updating tracking: {str(e)}"}

    def _calculate_delivery_window(
        self,
        shipping_method: str,
        estimated_date: timezone.datetime
    ) -> Dict:
        """Calculate delivery time window based on shipping method"""
        windows = {
            Order.ShippingMethod.STANDARD: {'start': 8, 'end': 20},
            Order.ShippingMethod.EXPRESS: {'start': 9, 'end': 18},
            Order.ShippingMethod.OVERNIGHT: {'start': 8, 'end': 12},
            Order.ShippingMethod.LOCAL: {'start': 10, 'end': 22},
        }

        window = windows.get(shipping_method, {'start': 8, 'end': 20})
        return {
            'start_time': f"{window['start']:02d}:00",
            'end_time': f"{window['end']:02d}:00",
        }

    def _calculate_delivery_confidence(self, order: Order) -> str:
        """Calculate confidence level in delivery estimate"""
        try:
            # Get tracking history
            tracking_count = order.tracking_history.count()
            latest_tracking = order.tracking_history.order_by(
                '-timestamp').first()

            # Base confidence on various factors
            if not latest_tracking:
                return "LOW"

            # Check if order is already delivered
            if latest_tracking.status == OrderTracking.TrackingStatus.DELIVERED:
                return "DELIVERED"

            # Check for delivery exceptions
            if latest_tracking.status == OrderTracking.TrackingStatus.EXCEPTION:
                return "LOW"

            # Calculate confidence based on shipping progress
            if tracking_count < 2:
                return "MEDIUM"

            # High confidence if package is out for delivery
            if latest_tracking.status == OrderTracking.TrackingStatus.OUT_FOR_DELIVERY:
                return "HIGH"

            # Check if making expected progress
            expected_statuses = len(OrderTracking.TrackingStatus.choices)
            current_status_index = list(
                OrderTracking.TrackingStatus).index(latest_tracking.status)
            progress = current_status_index / expected_statuses

            if progress > 0.7:
                return "HIGH"
            elif progress > 0.3:
                return "MEDIUM"
            else:
                return "LOW"

        except Exception as e:
            logger.error(f"Error calculating delivery confidence: {str(e)}")
            logger.error(traceback.format_exc())
            return "UNKNOWN"

    def _update_order_status_from_tracking(
        self,
        order: Order,
        tracking_status: str
    ) -> None:
        """Update order status based on tracking status"""
        # Map tracking statuses to order statuses
        status_mapping = {
            OrderTracking.TrackingStatus.ORDER_PLACED: Order.Status.PENDING,
            OrderTracking.TrackingStatus.PROCESSING: Order.Status.PROCESSING,
            OrderTracking.TrackingStatus.SHIPPED: Order.Status.SHIPPED,
            OrderTracking.TrackingStatus.IN_TRANSIT: Order.Status.IN_TRANSIT,
            OrderTracking.TrackingStatus.DELIVERED: Order.Status.DELIVERED,
            OrderTracking.TrackingStatus.RETURNED: Order.Status.RETURNED,
        }

        new_status = status_mapping.get(tracking_status)
        if new_status and order.status != new_status:
            order.status = new_status
            order.save(update_fields=['status', 'modified'])

    @database_sync_to_async
    def get_conversation_history(self, conversation_id, limit=4):
        """Fetch conversation history including all message types"""
        conversation = Conversation.objects.get(id=conversation_id)
        messages = []

        # Get messages in correct order
        for msg in conversation.messages.all().order_by('created')[:limit]:
            if msg.is_from_user:
                messages.append(HumanMessage(content=msg.user_text.content))
            else:
                # For AI messages, include tool calls if they exist
                if hasattr(msg.ai_text, 'tool_calls') and msg.ai_text.tool_calls:
                    messages.append(AIMessage(
                        content=msg.ai_text.content,
                        additional_kwargs={
                            'tool_calls': msg.ai_text.tool_calls}
                    ))
                else:
                    messages.append(AIMessage(content=msg.ai_text.content))

        return messages

    @database_sync_to_async
    def get_or_create_conversation(
        self,
        conversation_id: str,
        order_id: Optional[str] = None
    ) -> Tuple[Conversation, Optional[Order]]:
        """Get or create conversation with optional order link"""
        try:
            conversation, created = Conversation.objects.get_or_create(
                id=conversation_id,
                defaults={
                    'user': self.user,
                    'status': 'AC',
                    'title': 'Order Support Conversation'
                }
            )

            if order_id:
                order = Order.objects.get(id=order_id)
                OrderConversationLink.objects.get_or_create(
                    order=order,
                    conversation=conversation
                )

                # End other active conversations
                if created:
                    Conversation.objects.filter(
                        user=self.user,
                        status='AC'
                    ).exclude(
                        id=conversation_id
                    ).update(status='EN')

                return conversation, order

            return conversation, None

        except Exception as e:
            logger.error(f"Error in get_or_create_conversation: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    @database_sync_to_async
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID if it belongs to the user"""
        try:
            return Order.objects.get(id=order_id, user=self.user)
        except Order.DoesNotExist:
            return None

    @database_sync_to_async
    def link_order_to_conversation(self, order: Order, conversation: Conversation):
        """Link an order to a conversation if not already linked"""
        from orders.models import OrderConversationLink
        OrderConversationLink.objects.get_or_create(
            order=order,
            conversation=conversation
        )

    @database_sync_to_async
    def save_message(
        self,
        conversation_id: str,
        content_type: str = 'TE',
        is_from_user: bool = True,
        in_reply_to: Optional[Message] = None
    ) -> Message:
        """Save a message to the conversation"""
        conversation = Conversation.objects.get(id=conversation_id)
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )

    @database_sync_to_async
    def save_usertext(self, message: Message, content: str) -> UserText:
        """Save user message content"""
        return UserText.objects.create(
            message=message,
            content=content
        )

    @database_sync_to_async
    def save_aitext(
        self,
        message: Message,
        content: str,
        tool_calls: Optional[List] = None
    ) -> AIText:
        """Save AI message content with optional tool calls"""
        return AIText.objects.create(
            message=message,
            content=content,
            tool_calls=tool_calls if tool_calls else []
        )

    @database_sync_to_async
    def get_order_details(self, order_id: str) -> dict:
        """Fetch detailed order information"""
        try:
            order = Order.objects.get(id=order_id)
            return {
                "order_id": str(order.id),
                "status": order.get_status_display(),
                "status_description": order.status,
                "user": order.user,
                "items": [
                    {
                        "product_id": str(item.product.id),
                        "product_name": item.product.name,
                        "quantity": item.quantity,
                        "price": str(item.price)
                    }
                    for item in order.items.all()
                ],
                "total_amount": str(sum(item.price * item.quantity for item in order.items.all()))
            }
        except Order.DoesNotExist:
            return {"error": "Order not found"}

    @database_sync_to_async
    def validate_order_status_for_modification(self, order_id: str) -> tuple[bool, str]:
        """
        Validate if an order can be modified based on its status.
        Returns a tuple of (is_valid, message).
        """
        try:
            order = Order.objects.get(id=order_id, user=self.user)
            # Define valid statuses for modification
            MODIFIABLE_STATUSES = ['PE', 'PR']
            if order.status not in MODIFIABLE_STATUSES:
                status_display = order.get_status_display()
                return False, f"Cannot modify order in {status_display} status. Only pending or processing orders can be modified."

            return True, "Order is eligible for modification"

        except Order.DoesNotExist:
            return False, "Order not found or access denied"
        except Exception as e:
            logger.error(f"Error validating order status: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"Error validating order status: {str(e)}"

    async def update_order(self, order_id, update_data):
        try:
            # First validate order status
            validation_result = await self.validate_order_status_for_modification(order_id)
            can_modify, message = validation_result
            if not can_modify:
                raise ValueError(message)

            # Proceed with update using database_sync_to_async
            @database_sync_to_async
            def perform_update():
                with transaction.atomic():
                    # Lock the order and related records
                    order = Order.objects.select_for_update().get(id=order_id)

                    # Validate user permissions
                    if order.user != self.user:
                        raise PermissionError(
                            "Not authorized to modify this order")

                    action = update_data.get('action')

                    if action == 'modify_quantity':
                        return self._handle_quantity_modification(order, update_data)
                    elif action == 'cancel_item':
                        return self._handle_item_cancellation(order, update_data)
                    elif action == 'cancel_order':
                        return self._handle_order_cancellation(order)
                    else:
                        raise ValueError(f"Unknown action: {action}")

            # Execute the update
            result = await perform_update()
            return result

        except (Order.DoesNotExist, OrderItem.DoesNotExist):
            raise ValueError("Order or item not found")
        except Exception as e:
            logger.error(f"Error updating order: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _handle_quantity_modification(self, order, update_data):
        # Get and lock the specific order item
        order_item = OrderItem.objects.select_for_update().get(
            product__id=update_data['item_id'],
            order=order
        )

        # Validate stock availability
        if update_data['new_quantity'] > order_item.product.stock:
            raise ValueError(
                f"Insufficient stock. Only {order_item.product.stock} available.")

        # Store old quantity for price adjustment
        old_quantity = order_item.quantity

        # Update quantity and price
        order_item.quantity = update_data['new_quantity']
        order_item.price = order_item.product.price * \
            update_data['new_quantity']
        order_item.save()

        # Update order total
        order.total_amount = sum(
            item.price for item in order.items.all()
        )
        order.save()

        return {
            'status': 'success',
            'message': f"Quantity updated from {old_quantity} to {update_data['new_quantity']}",
            'new_total': order.total_amount
        }

    def _handle_item_cancellation(self, order, update_data):
        # Get and lock the specific order item
        order_item = OrderItem.objects.select_for_update().get(
            product__id=update_data['item_id'],
            order=order
        )

        # Store item details for confirmation message
        item_name = order_item.product.name
        item_quantity = order_item.quantity

        # Delete the item
        order_item.delete()

        # Update order total
        order.total_amount = sum(
            item.price for item in order.items.all()
        )

        # If no items left, mark order as cancelled
        if not order.items.exists():
            order.status = Order.Status.CANCELLED

        order.save()

        return {
            'status': 'success',
            'message': f"Removed {item_quantity}x {item_name} from order",
            'new_total': order.total_amount
        }

    def _handle_order_cancellation(self, order):
        # Can only cancel if order is in certain states
        if order.status not in [Order.Status.PENDING, Order.Status.PROCESSING]:
            raise ValueError(
                "Order cannot be cancelled in its current state")

        order.status = Order.Status.CANCELLED
        order.save()

        return {
            'status': 'success',
            'message': f"Order #{order.id} has been cancelled",
            'new_status': order.get_status_display()
        }

    @database_sync_to_async
    def render_order_card(self, order_details):
        """
        Render updated order card HTML for e-commerce order
        Includes order details, items, status, and total
        """

        items_html = ""
        for item in order_details['items']:
            items_html += f"""
                <div class="order-item mb-2">
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Product:</strong> {str(item['product_name'])}
                        </div>
                        <div class="col-md-2">
                            <strong>Qty:</strong> {str(item['quantity'])}
                        </div>
                        <div class="col-md-4">
                            <strong>Price:</strong> ${str(item['price'])}
                        </div>
                    </div>
                </div>
            """

        status_class = {
            'PE': 'text-warning',  # Pending
            'PR': 'text-info',     # Processing
            'SH': 'text-primary',  # Shipped
            'DE': 'text-success',  # Delivered
            'CA': 'text-danger',   # Cancelled
            'RT': 'text-secondary'  # Returned
        }.get(order_details['status'], '')

        return f"""
        <div class="card mb-3" data-order-id="{str(order_details['order_id'])}">
            <div class="card-header">
                <div class="row">
                    <div class="col-md-2">
                        <span class="text-body-secondary">ORDER ID</span><br>
                        Order #{str(order_details['order_id'])}
                    </div>
                    <div class="col-md-2">
                        <span class="text-body-secondary">ORDER PLACED</span><br>
                        {order_details['status']}
                    </div>
                    <div class="col-md-2">
                        <span class="text-body-secondary">TOTAL</span><br>
                        ${str(order_details['total_amount'])}
                    </div>
                    <div class="col-md-4">
                        <span class="text-body-secondary">STATUS</span><br>
                        <span class="{status_class}">{order_details['status_description']}</span>
                    </div>
                    <div class="col-md-2">
                        <span class="text-body-secondary">SHIP TO</span><br>
                        {order_details['user']}
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="order-items">
                    {items_html}
                </div>
            </div>
        </div>
        """

    @database_sync_to_async
    def save_conversation_title(conversation, title):
        conversation.title = title
        conversation.save()

    @database_sync_to_async
    def _extract_new_quantity(self, content: str) -> Optional[int]:
        """Extract the new quantity from the user's message"""
        try:
            import re
            # Look for patterns like "change quantity to X" or "to X"
            quantity_patterns = [
                r'change (?:the )?quantity to (\d+)',
                r'quantity to (\d+)',
                r'change (?:it )?to (\d+)',
                r'to (\d+)',
                r'quantity (?:of )?(\d+)'
            ]

            for pattern in quantity_patterns:
                matches = re.search(pattern, content.lower())
                if matches:
                    return int(matches.group(1))

            return None
        except Exception as e:
            logger.error(f"Error extracting quantity: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @database_sync_to_async
    def get_recent_orders(self, user_id: int, limit: int = 5) -> List[Dict]:
        """Get user's recent orders"""
        recent_orders = Order.objects.filter(
            user_id=user_id
        ).order_by('-created')[:limit]

        return [{
            'id': order.id,
            'created_date': order.created.strftime("%Y-%m-%d"),
            'status': order.get_status_display(),
            'item_count': order.items.count(),
            'total_amount': order.total_amount
        } for order in recent_orders]

    @database_sync_to_async
    def get_or_create_general_conversation(self, conversation_id: str) -> Conversation:
        """Create or get a general conversation without order context"""
        conversation, created = Conversation.objects.get_or_create(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC',
                'title': 'General Support Conversation'
            }
        )

        if created:
            # End other active conversations
            Conversation.objects.filter(
                user=self.user,
                status='AC'
            ).exclude(id=conversation_id).update(status='EN')

        return conversation
