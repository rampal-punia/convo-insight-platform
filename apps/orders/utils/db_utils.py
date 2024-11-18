from channels.db import database_sync_to_async
import traceback
from django.db import transaction
from convochat.models import Conversation, Message, UserText, AIText
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from ..models import Order, OrderConversationLink, OrderItem
import logging
logger = logging.getLogger('orders')  # Get the orders logger


class DatabaseOperations:
    def __init__(self, user):
        self.user = user

    @database_sync_to_async
    def get_conversation_history(self, conversation_id, limit=8):
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

    async def get_or_create_conversation(self, conversation_id, order_id):
        """Create or get existing conversation and link it to the order"""
        conversation, created = await database_sync_to_async(Conversation.objects.update_or_create)(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )

        order = await database_sync_to_async(Order.objects.get)(id=order_id)
        await database_sync_to_async(OrderConversationLink.objects.get_or_create)(
            order=order,
            conversation=conversation
        )

        if created:
            await database_sync_to_async(
                Conversation.objects.filter(user=self.user, status='AC').exclude(
                    id=conversation_id).update
            )(status='EN')

        return conversation, order

    @database_sync_to_async
    def save_message(self, conversation, content_type, is_from_user=True, in_reply_to=None):
        """Save a message to the database"""
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )

    @database_sync_to_async
    def save_usertext(self, message, input_data):
        """Save user text content"""
        return UserText.objects.create(
            message=message,
            content=input_data,
        )

    @database_sync_to_async
    def save_aitext(self, message, input_data, tool_calls=None):
        """Save AI text content with optional tool calls"""
        return AIText.objects.create(
            message=message,
            content=input_data,
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
    def update_order(self, order_id, update_data):
        try:
            with transaction.atomic():
                # Lock the order and related records
                order = Order.objects.select_for_update().get(id=order_id)
                # Validate order status
                if order.status not in ['PE', 'PR']:
                    return "Cannot modify order - it has already been shipped or delivered."

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
