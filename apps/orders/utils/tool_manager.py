from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import traceback
from enum import Enum

from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=2)

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


class BaseOrderSchema(BaseModel):
    """Base schema for order-related tools"""
    order_id: str = Field(description="The ID of the order")
    customer_id: str = Field(description="The ID of the customer")


class TrackingRequest(BaseModel):
    """Base schema for tracking-related tools"""
    order_id: str = Field(description="The ID of the order to track")
    customer_id: str = Field(
        description="The ID of the customer making the request")


class TrackingDetailRequest(TrackingRequest):
    """Schema for detailed tracking information"""
    include_history: bool = Field(
        default=False,
        description="Whether to include full tracking history"
    )


class ShipmentLocationRequest(TrackingRequest):
    """Schema for current shipment location"""
    pass


class DeliveryEstimateRequest(TrackingRequest):
    """Schema for delivery estimate information"""
    pass


class TrackingUpdateRequest(TrackingRequest):
    """Schema for tracking updates subscription"""
    notify: bool = Field(
        default=True,
        description="Whether to enable tracking notifications"
    )


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
            'query': ['get_order_info', 'track_order', 'get_shipment_location'],
            'tracking': ['get_tracking_details', 'get_delivery_estimate'],
            'status': ['track_order', 'get_order_info'],
            'modification': ['modify_order_quantity'],
            'cancellation': ['cancel_order']
        }

    def register(self, tool: BaseTool, metadata: ToolMetadata):
        """Register a tool with its metadata"""
        # Prevent registration of redundant tools
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} is already registered")
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
        self.tool_call_depth = 0  # Add this line
        self.MAX_TOOL_DEPTH = 3   # Add this line
        self.current_tool_chain = []  # Add this line

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

        # Register tracking-specific tools
        self.registry.register(
            self._create_tracking_details_tool(),
            ToolMetadata(
                category=ToolCategory.SAFE,
                description="Get detailed tracking information",
                requires_confirmation=False
            )
        )

        self.registry.register(
            self._create_shipment_location_tool(),
            ToolMetadata(
                category=ToolCategory.SAFE,
                description="Get current shipment location",
                requires_confirmation=False
            )
        )

        self.registry.register(
            self._create_delivery_estimate_tool(),
            ToolMetadata(
                category=ToolCategory.SAFE,
                description="Get estimated delivery date and time",
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
        """Create the main tracking tool"""
        @tool(args_schema=TrackingRequest)
        async def track_order(order_id: str, customer_id: str) -> str:
            """Get comprehensive tracking information for an order"""
            try:
                # Check recursion depth
                if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
                    return "Unable to process request due to too many nested tool calls"

                self.tool_call_depth += 1
                self.current_tool_chain.append('track_order')

                try:
                    # Get all required information in one go
                    order_details = await self.db_ops.get_order_details(order_id)
                    if 'error' in order_details:
                        return "Order not found"

                    tracking_info = await self.db_ops.get_tracking_info(order_id)

                    # Format response
                    response = (
                        f"Order #{order_details['order_id']}\n"
                        f"Status: {order_details['status']}\n"
                        f"Tracking Number: {tracking_info['tracking_number']}\n"
                        f"Carrier: {tracking_info['carrier']}\n"
                        f"Last Location: {tracking_info['current_location']}\n"
                        f"Estimated Delivery: {tracking_info['estimated_delivery']}\n\n"
                        f"Latest Update: {tracking_info['latest_update']}"
                    )

                    return response
                finally:
                    self.tool_call_depth -= 1
                    self.current_tool_chain.pop()

            except Exception as e:
                logger.error(f"Error tracking order: {str(e)}")
                return "Failed to retrieve tracking information"
        return track_order

    def _create_tracking_details_tool(self):
        """Create tool for detailed tracking information"""
        @tool(args_schema=TrackingDetailRequest)
        async def get_tracking_details(
            order_id: str,
            customer_id: str,
            include_history: bool = False
        ) -> str:
            """Get detailed tracking information with optional history"""
            # Check recursion depth
            if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
                return "Unable to process request due to too many nested tool calls"

            # Check if this tool is already in the chain
            if 'get_tracking_details' in self.current_tool_chain:
                return "Already processing tracking details"

            self.tool_call_depth += 1
            self.current_tool_chain.append('get_tracking_details')

            try:
                tracking_info = await self.db_ops.get_tracking_info(order_id)

                if include_history:
                    history = await self.db_ops.get_tracking_history(order_id)
                    tracking_info['history'] = history

                return self._format_tracking_details(tracking_info)
            finally:
                self.tool_call_depth -= 1
                self.current_tool_chain.pop()
        return get_tracking_details

    def _create_shipment_location_tool(self):
        """Create tool for current shipment location"""
        @tool(args_schema=ShipmentLocationRequest)
        async def get_shipment_location(order_id: str, customer_id: str) -> str:
            """Get current location and status of the shipment"""
            if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
                return "Unable to process request due to too many nested tool calls"

            if 'get_shipment_location' in self.current_tool_chain:
                return "Already processing location information"

            self.tool_call_depth += 1
            self.current_tool_chain.append('get_shipment_location')

            try:
                location_info = await self.db_ops.get_current_location(order_id)
                return (
                    f"Current Location: {location_info['location']}\n"
                    f"Status: {location_info['status']}\n"
                    f"Last Updated: {location_info['timestamp']}"
                )
            finally:
                self.tool_call_depth -= 1
                self.current_tool_chain.pop()
        return get_shipment_location

    def _create_delivery_estimate_tool(self):
        """Create tool for delivery estimates"""
        @tool(args_schema=DeliveryEstimateRequest)
        async def get_delivery_estimate(order_id: str, customer_id: str) -> str:
            """Get estimated delivery date and time window"""
            if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
                return "Unable to process request due to too many nested tool calls"

            if 'get_delivery_estimate' in self.current_tool_chain:
                return "Already processing delivery estimate"

            self.tool_call_depth += 1
            self.current_tool_chain.append('get_delivery_estimate')

            try:
                estimate = await self.db_ops.get_delivery_estimate(order_id)
                return (
                    f"Estimated Delivery: {estimate['date']}\n"
                    f"Time Window: {estimate['time_window']}\n"
                    f"Confidence: {estimate['confidence']}"
                )
            finally:
                self.tool_call_depth -= 1
                self.current_tool_chain.pop()
        return get_delivery_estimate

    def is_sensitive_tool(self, tool_name: str) -> bool:
        """Check if a tool is sensitive"""
        if not tool_name:
            logger.warning("Received empty tool name for sensitivity check")
            return False

        # Check tool chain depth
        if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
            logger.warning(f"Tool chain depth exceeded for {tool_name}")
            return True  # Treat as sensitive to prevent further calls

        metadata = self.registry.get_tool_metadata(tool_name)
        is_sensitive = (metadata is not None and
                        metadata.category == ToolCategory.SENSITIVE)
        logger.info(f"Tool {tool_name} is_sensitive: {is_sensitive}")
        return is_sensitive

    def _format_tracking_details(self, tracking_info: Dict) -> str:
        """Format tracking information into a readable string"""
        details = (
            f"Tracking Details for Order #{tracking_info['order_id']}\n"
            f"Status: {tracking_info['status']}\n"
            f"Carrier: {tracking_info['carrier']}\n"
            f"Tracking Number: {tracking_info['tracking_number']}\n\n"
            f"Current Location: {tracking_info['current_location']}\n"
            f"Last Updated: {tracking_info['last_update']}\n"
            f"Estimated Delivery: {tracking_info['estimated_delivery']}\n"
        )

        if 'history' in tracking_info:
            details += "\nTracking History:\n"
            for event in tracking_info['history']:
                details += f"- {event['timestamp']}: {event['status']} - {event['location']}\n"

        return details

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


def create_tool_manager(db_ops) -> ToolManager:
    """Factory function to create a tool manager instance"""
    return ToolManager(db_ops)
