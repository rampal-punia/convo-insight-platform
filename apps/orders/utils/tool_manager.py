import logging
from enum import Enum
from dataclasses import dataclass
import traceback
from typing import Annotated, TypedDict, List, Dict, Optional
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger('orders')


class ToolCategory(Enum):
    """Enum to categorize tools"""
    SAFE = "safe"
    SENSITIVE = "sensitive"


@dataclass
class ToolMetadata:
    """Metadata for tool categorization and configuration"""
    category: ToolCategory
    description: str
    requires_confirmation: bool = False


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
    completed: bool


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


class ToolRegistry:
    """Registry to manage and categorize tools"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.tool_categories = {
            'query': ['get_order_info'],  # Information retrieval tools
            # Order modification tools
            'modification': ['modify_order_quantity'],
            'status': ['track_order'],  # Status check tools
            'cancellation': ['cancel_order']  # Cancellation tools
        }

    def register(self, tool: BaseTool, metadata: ToolMetadata):
        """Register a tool with its metadata"""
        # Prevent registration of redundant tools
        for category, tools in self.tool_categories.items():
            if tool.name in tools:
                existing_tool = next(
                    (t for t in self.tools.values() if t.name in tools), None)
                if existing_tool:
                    logger.warning(
                        f"Tool {tool.name} conflicts with existing tool {existing_tool.name} in category {category}")
                    return

        self.tools[tool.name] = tool
        self.metadata[tool.name] = metadata

    def tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools of a specific category"""
        return [
            tool for tool_name, tool in self.tools.items()
            if self.metadata[tool_name].category == category
        ]

    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool"""
        return self.metadata.get(tool_name)

    def is_sensitive_tool(self, tool_name: str) -> bool:
        """Check if a tool is sensitive"""
        if not tool_name:
            logger.warning("Received empty tool name for sensitivity check")
            return False

        metadata = self.get_tool_metadata(tool_name)
        is_sensitive = (metadata is not None and
                        metadata.category == ToolCategory.SENSITIVE)
        logger.info(f"Tool {tool_name} is_sensitive: {is_sensitive}")
        return is_sensitive


class ToolManager:
    """Manager class for tool operations and categorization"""

    def __init__(self, db_ops):
        self.db_ops = db_ops
        self.registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self):
        """Register all tools with their metadata"""
        # Register safe tools
        self.registry.register(
            self._create_track_order_tool(),
            ToolMetadata(
                category=ToolCategory.SAFE,
                description="Track order status",
                requires_confirmation=False
            )
        )

        self.registry.register(
            self._create_get_order_info_tool(),
            ToolMetadata(
                category=ToolCategory.SAFE,
                description="Get support information",
                requires_confirmation=False
            )
        )

        # Register sensitive tools
        self.registry.register(
            self._create_modify_order_tool(),
            ToolMetadata(
                category=ToolCategory.SENSITIVE,
                description="Modify order quantities",
                requires_confirmation=True
            )
        )

        self.registry.register(
            self._create_cancel_order_tool(),
            ToolMetadata(
                category=ToolCategory.SENSITIVE,
                description="Cancel entire order",
                requires_confirmation=True
            )
        )

    def _create_track_order_tool(self):
        """Create the track order tool"""
        @tool(args_schema=TrackOrder)
        async def track_order(order_id: str, customer_id: str) -> str:
            """Get tracking information for an order"""
            try:
                order_details = await self.db_ops.get_order_details(order_id)
                if 'error' in order_details:
                    return "Order not found"

                return (
                    f"Order #{order_details['order_id']}\n"
                    f"Status: {order_details['status']}\n"
                    f"Items:\n" +
                    "\n".join([
                        f"- {item['quantity']}x {item['product']}"
                        for item in order_details['items']
                    ])
                )
            except Exception as e:
                logger.error(f"Error tracking order: {str(e)}")
                logger.error(traceback.format_exc())
                return "Failed to get tracking information"
        return track_order

    def _create_modify_order_tool(self):
        """Create the modify order tool"""
        @tool(args_schema=ModifyOrderQuantity)
        async def modify_order_quantity(
            order_id: str,
            customer_id: str,
            product_id: int,
            new_quantity: int
        ) -> str:
            """Modify the quantity of a product in an order"""
            # First validate order status
            try:
                can_modify, message = await self.db_ops.validate_order_status_for_modification(order_id)
                if not can_modify:
                    logger.warning(f"Order modification rejected: {message}")
                    return message

                logger.info(f"Modifying order quantity: {new_quantity}")
                result = await self.db_ops.update_order(
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
                logger.error(traceback.format_exc())
                return "Failed to modify order quantity"
        return modify_order_quantity

    def _create_cancel_order_tool(self):
        """Create the cancel order tool"""
        @tool(args_schema=CancelOrder)
        async def cancel_order(order_id: str, customer_id: str, reason: str) -> str:
            """Cancel an order if eligible"""
            try:
                result = await self.db_ops.update_order(
                    order_id,
                    {
                        'action': 'cancel_order',
                        'reason': reason
                    }
                )
                return result['message']
            except Exception as e:
                logger.error(f"Error cancelling order: {str(e)}")
                logger.error(traceback.format_exc())
                return "Failed to cancel order"
        return cancel_order

    def _create_get_order_info_tool(self):
        """Create the get support info tool"""
        @tool(args_schema=GetSupportInfo)
        async def get_order_info(order_id: str, customer_id: str) -> str:
            """Get support information and available actions"""
            try:
                order_details = await self.db_ops.get_order_details(order_id)
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
                logger.error(traceback.format_exc())
                return "Failed to get support information"
        return get_order_info

    def get_safe_tools(self) -> List[BaseTool]:
        """Get all safe tools"""
        return self.registry.tools_by_category(ToolCategory.SAFE)

    def get_sensitive_tools(self) -> List[BaseTool]:
        """Get all sensitive tools"""
        return self.registry.tools_by_category(ToolCategory.SENSITIVE)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all tools"""
        return list(self.registry.tools.values())

    def is_sensitive_tool(self, tool_name: str) -> bool:
        """Check if a tool is sensitive"""
        return self.registry.is_sensitive_tool(tool_name)


def create_tool_manager(db_ops) -> ToolManager:
    """Factory function to create a tool manager instance"""
    return ToolManager(db_ops)
