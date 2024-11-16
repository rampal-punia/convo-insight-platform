from typing import Annotated, TypedDict, List, Dict

from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from channels.db import database_sync_to_async

from ..models import Order, OrderItem


class OrderState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_info: dict
    order_info: dict
    cart: dict
    intent: str
    conversation_id: str
    modified: bool
    confirmation_pending: bool


class SummarizeConversation(BaseModel):
    '''Tool for summarizing the current conversation state.'''
    conversation_id: str = Field(
        description="The ID of the conversation to summarize")
    max_length: int = Field(
        default=150,
        description="Maximum length of the summary in words"
    )


class ReturnRequest(BaseModel):
    """Schema for return requests"""
    order_id: str
    items: List[dict] = Field(
        description="List of items to return"
    )
    reason: str = Field(
        description="Reason for return"
    )
    condition: str = Field(
        description="Condition of items being returned"
    )


class ModifyOrderQuantity(BaseModel):
    '''Tools to modify the quantity of items in an order.'''
    order_id: int = Field(description="The ID of the order to modify")
    product_id: int = Field(description="The ID of the product to modify")
    new_quantity: int = Field(description="The new quantity desired")


class CancelOrder(BaseModel):
    '''Tool to cancel an entire order'''
    order_id: int = Field(description="The ID of the order to cancel")
    reason: str = Field(description="The reason for cancellation")


class TrackOrder(BaseModel):
    '''Tool to get detailed tracking information for an order.'''
    order_id: int = Field(description="The ID of the order to track")


class GetSupportInfo(BaseModel):
    '''Tool to get support information and eligible actions for an order.'''
    order_id: int = Field(description="The ID of the order to get support for")


def create_order_tools(order_id: int):
    @tool
    async def get_read_only_tools() -> list:
        """Read-only tools that don't require confirmation"""
        return [
            get_order_details,
            #    check_order_status,
            #    search_products,
            #    get_return_policy
        ]

    @tool
    async def get_order_details(order_id: str, *, config) -> dict:
        """Fetch detailed order information using Django models."""
        try:
            # Get order with related items
            order = Order.objects.select_related('user').prefetch_related(
                'items__product'
            ).get(
                id=order_id,
                user_id=config["customer_id"]
            )

            # Structure the response
            return {
                "order_id": str(order.id),
                "status": order.get_status_display(),
                "items": [
                    {
                        "product_id": str(item.product.id),
                        "name": item.product.name,
                        "quantity": item.quantity,
                        "price": str(item.price)
                    }
                    for item in order.items.all()
                ],
                "total": str(sum(item.price * item.quantity for item in order.items.all()))
            }
        except Order.DoesNotExist:
            return {"error": "Order not found"}

    @tool
    async def modify_order_quantity(order_id: int, product_id: int, new_quantity: int) -> str:
        '''Modify the quantity of a product in an order.'''
        # try:
        order = await database_sync_to_async(Order.objects.get)(id=order_id)
        order_item = await database_sync_to_async(OrderItem.objects.get)(order=order, product__id=product_id)

        if order.status not in ['PE', 'PR']:
            return "Cannot modify order - it has already been shipped or delivered."

        product = order_item.product
        if new_quantity > product.stock:
            return f"Cannot modify - only {product.stock} units available in stock."

        old_quantity = order_item.quantity
        order_item.quantity = new_quantity
        order_item.price = product.price * new_quantity
        await database_sync_to_async(order_item.save)()

        # Update total amount
        order.total_amount = await database_sync_to_async(sum)(
            [item.price for item in order.items.all()]
        )
        await database_sync_to_async(order.save)()

        return f"Successfully updated quantity from {old_quantity} to {new_quantity}."

        # except Order.DoesNotExist:
        #     return f"Order {order_id} not found."
        # except OrderItem.DoesNotExist:
        #     return f"Product {product_id} not found in order {order_id}."

    @tool
    async def cancel_order(order_id: int, reason: str) -> str:
        '''Cancel an order if it hasn't been shipped'''
        try:
            order = await database_sync_to_async(Order.objects.get)(id=order_id)

            if order.status not in ['PE', 'PR']:
                return "Cannot cancel order - it has already been shipped or delivered."

            order.status = 'CA'
            await database_sync_to_async(order.save)()
            return f"Order {order_id} successfully cancelled. Reason: {reason} "

        except Order.DoesNotExist:
            return f"Order {order_id} not found."

    @tool
    async def track_order(order_id: int) -> str:
        '''Get detailed tracking information for an order.'''
        try:
            order = await database_sync_to_async(Order.objects.get)(id=order_id)
            status_desc = order.get_status_display()
            items = await database_sync_to_async(list)(order.items.all())

            tracking_info = (
                f"Order #{order.id}\n"
                f"Status: #{status_desc}\n"
                f"Created: #{order.created}\n"
                f"Items: \n"
            )
            for item in items:
                tracking_info += f"- {item.quantity}x {item.product.name}\n"

            return tracking_info

        except Order.DoesNotExist:
            return f"Order {order_id} not found."

    @tool
    async def get_support_info(order_id: int) -> str:
        '''Get support information and eligible actions for an order.'''
        try:
            order = await database_sync_to_async(Order.objects.get)(id=order_id)

            info = (
                f"Order #{order.id} Support Information\n"
                f"Current Status: {order.get_status_display()}\n"
                f"Order Date: {order.created}\n\n"
                "Available Actions:\n"
            )

            if order.status in ['PE', 'PR']:
                info += "- Modify quantities\n- Cancel order\n"
            elif order.status in ['SH', 'TR']:
                info += "- Track shipment\n"
            elif order.status in ['DE']:
                info += "- Return items (within 30 days of delivery)\n"

            return info

        except Order.DoesNotExist:
            return f"Order {order_id} not found."

    return [
        modify_order_quantity,
        cancel_order,
        track_order,
        get_support_info
    ]
