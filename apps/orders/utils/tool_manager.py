import logging

from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from .db_utils import DatabaseOperations

logger = logging.getLogger('orders')


class OrderState(TypedDict):
    """Represents the state of an order in the system"""
    messages: Annotated[list[AnyMessage], add_messages]
    customer_info: dict
    order_info: dict
    cart: dict
    intent: str
    conversation_id: str
    modified: bool
    confirmation_pending: bool


class BaseOrderSchema(BaseModel):
    """Base schema for order-related tools"""
    order_id: str = Field(description="The ID of the order")
    customer_id: str = Field(description="The ID of the customer")


class ModifyOrderQuantity(BaseOrderSchema):
    '''Tools to modify the quantity of items in an order.'''
    product_id: int = Field(description="The ID of the product to modify")
    new_quantity: int = Field(description="The new quantity desired")


class CancelOrder(BaseOrderSchema):
    """Schema for order cancellation"""
    reason: str = Field(description="The reason for cancellation")


class TrackOrder(BaseOrderSchema):
    """Schema for order tracking"""
    pass


class GetSupportInfo(BaseOrderSchema):
    """Schema for getting support information"""
    pass


class ReturnRequest(BaseOrderSchema):
    """Schema for return requests"""
    items: List[dict] = Field(description="List of items to return")
    reason: str = Field(description="Reason for return")
    condition: str = Field(description="Condition of items being returned")


class SummarizeConversation(BaseModel):
    """Schema for conversation summarization"""
    conversation_id: str = Field(
        description="The ID of the conversation to summarize")
    max_length: int = Field(
        default=150,
        description="Maximum length of the summary in words"
    )


def tool_manager(db_ops):
    @tool(args_schema=BaseOrderSchema)
    async def get_order_details(order_id: str, customer_id: str) -> dict:
        """Fetch detailed order information"""
        try:
            return await db_ops.get_order_details(order_id)
        except Exception as e:
            logger.error(f"Error getting order details: {str(e)}")
            return {"error": "Failed to fetch order details"}

    @tool(args_schema=ModifyOrderQuantity)
    async def modify_order_quantity(order_id: str, customer_id: str, product_id: int, new_quantity: int) -> str:
        """Modify the quantity of a product in an order"""
        try:
            # Use DatabaseOperations for the modification
            result = await db_ops.update_order(
                order_id,
                {
                    'action': 'modify_quantity',
                    'item_id': product_id,
                    'new_quantity': new_quantity
                }
            )
            return result['message']
        except Exception as e:
            logger.error(f"Error modifying order quantity: {str(e)}")
            return "Failed to modify order quantity"

    @tool(args_schema=CancelOrder)
    async def cancel_order(order_id: str, customer_id: str, reason: str) -> str:
        """Cancel an order if eligible"""
        try:
            result = await db_ops.update_order(
                order_id,
                {
                    'action': 'cancel_order',
                    'reason': reason
                }
            )
            return result['message']
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return "Failed to cancel order"

    @tool(args_schema=TrackOrder)
    async def track_order(order_id: str, customer_id: str) -> str:
        """Get tracking information for an order"""
        try:
            order_details = await db_ops.get_order_details(order_id)
            if 'error' in order_details:
                return "Order not found"

            return (
                f"Order #{order_details['order_id']}\n"
                f"Status: {order_details['status']}\n"
                f"Items:\n" +
                "\n".join([
                    f"- {item['quantity']}x {item['name']}"
                    for item in order_details['items']
                ])
            )
        except Exception as e:
            logger.error(f"Error tracking order: {str(e)}")
            return "Failed to get tracking information"

    @tool(args_schema=GetSupportInfo)
    async def get_support_info(order_id: str, customer_id: str) -> str:
        """Get support information and available actions"""
        try:
            order_details = await db_ops.get_order_details(order_id)
            if 'error' in order_details:
                return "Order not found"

            status_actions = {
                'Pending': ["- Modify quantities", "- Cancel order"],
                'Processing': ["- Modify quantities", "- Cancel order"],
                'Shipped': ["- Track shipment"],
                'In Transit': ["- Track shipment"],
                'Delivered': ["- Return items (within 30 days of delivery)"]
            }

            return (
                f"Order #{order_details['order_id']} Support Information\n"
                f"Current Status: {order_details['status']}\n\n"
                "Available Actions:\n" +
                "\n".join(status_actions.get(order_details['status'], []))
            )
        except Exception as e:
            logger.error(f"Error getting support info: {str(e)}")
            return "Failed to get support information"

    def create_tools() -> List[BaseTool]:
        """Creates and returns all available tools"""
        return [
            get_order_details,
            modify_order_quantity,
            cancel_order,
            track_order,
            get_support_info
        ]

    return create_tools
