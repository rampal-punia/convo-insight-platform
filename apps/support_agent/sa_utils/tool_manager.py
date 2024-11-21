from langchain_core.tools import BaseTool
from typing import Any, Callable, List, cast, Dict
import logging
import traceback

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated
from .configuration import Configuration

logger = logging.getLogger('orders')


@tool
async def duckduckgo_search(
        query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> List[Dict[str, str]]:
    """Search for general web results using DuckDuckGo
    This function performs a search using the DuckDuckGo search engine, which is known
    for its privacy-focused approach and unbiased results.

    Args:
        query: The search query string
        config: Configuration including max_search_results

    Returns:
        List of search results, each containing title, link, and snippet
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize DuckDuckGo search wrapper with configuration
    search = DuckDuckGoSearchResults(
        output_format="list",
        num_results=configuration.max_search_results,
    )
    # Perform the search
    result = await search.ainvoke(query)
    return cast(list[dict[str, Any]], result)


@tool
async def track_order(
    order_id: str,
    customer_id: str,
    include_history: bool = False,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Get tracking details for a specific order.

    Args:
        order_id: The ID of the order to track
        customer_id: The ID of the customer making the request
        include_history: Whether to include full tracking history
        config: Configuration containing database operations

    Returns:
        Formatted string containing tracking information
    """
    try:
        db_ops = config.get("db_ops")
        if not db_ops:
            raise ValueError("Database operations not provided in config")

        # Get order details
        order_details = await db_ops.get_order_details(order_id)
        if not order_details:
            return "Order not found"

        # Get tracking information
        tracking_info = await db_ops.get_tracking_info(order_id)

        # Format basic response
        response = (
            f"Order #{order_details['order_id']}\n"
            f"Status: {order_details['status']}\n"
            f"Tracking Number: {tracking_info['tracking_number']}\n"
            f"Carrier: {tracking_info['carrier']}\n"
            f"Current Location: {tracking_info['current_location']}\n"
            f"Estimated Delivery: {tracking_info['estimated_delivery']}"
        )

        # Add history if requested
        if include_history and tracking_info.get('history'):
            response += "\n\nTracking History:"
            for event in tracking_info['history']:
                response += f"\n- {event['timestamp']}: {event['status']} at {event['location']}"

        return response

    except Exception as e:
        logger.error(f"Error in track_order: {str(e)}")
        logger.error(traceback.format_exc())
        return "Failed to retrieve tracking information"


@tool
async def modify_order_quantity(
    order_id: str,
    customer_id: str,
    product_id: int,
    new_quantity: int,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Modify the quantity of a product in an order.

    Args:
        order_id: The ID of the order to modify
        customer_id: The ID of the customer making the request
        product_id: The ID of the product to modify
        new_quantity: The new quantity desired
        config: Configuration containing database operations

    Returns:
        Message indicating the result of the modification
    """
    try:
        db_ops = config.get("db_ops")
        if not db_ops:
            raise ValueError("Database operations not provided in config")

        # Validate order status
        can_modify, message = await db_ops.validate_order_status_for_modification(order_id)
        if not can_modify:
            return message

        # Perform modification
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
        logger.error(f"Error in modify_order_quantity: {str(e)}")
        logger.error(traceback.format_exc())
        return "Failed to modify order quantity"


@tool
async def cancel_order(
    order_id: str,
    customer_id: str,
    reason: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Cancel an order with a specified reason.

    Args:
        order_id: The ID of the order to cancel
        customer_id: The ID of the customer making the request
        reason: The reason for cancellation
        config: Configuration containing database operations

    Returns:
        Message indicating the result of the cancellation
    """
    try:
        db_ops = config.get("db_ops")
        if not db_ops:
            raise ValueError("Database operations not provided in config")

        result = await db_ops.update_order(
            order_id,
            {
                'action': 'cancel_order',
                'reason': reason
            }
        )

        return result['message']

    except Exception as e:
        logger.error(f"Error in cancel_order: {str(e)}")
        logger.error(traceback.format_exc())
        return "Failed to cancel order"

# List of all available tools
TOOLS: List[Callable[..., Any]] = [
    duckduckgo_search,
    track_order,
    modify_order_quantity,
    cancel_order
]

# Tool sensitivity configuration (for use in consumers)
SENSITIVE_TOOLS = {
    "modify_order_quantity": True,
    "cancel_order": True
}


def get_sensitive_tool_names():
    return list(SENSITIVE_TOOLS.keys())


def is_sensitive_tool(tool_name: str) -> bool:
    """Check if a tool requires special handling."""
    return SENSITIVE_TOOLS.get(tool_name, False)


def get_all_tools() -> List[BaseTool]:
    """Get all available tools."""
    return TOOLS


if __name__ == '__main__':
    import asyncio

    async def test_searches():
        query = "what is artificial intelligence?"
        print("\nTesting DuckDuckGo Search:")
        ddg_results = await duckduckgo_search(query, config={})
        print(f"Found {len(ddg_results)} results")

    asyncio.run(test_searches())
