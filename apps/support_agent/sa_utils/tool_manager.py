from decimal import Decimal
from typing import Any, Callable, List, cast, Dict, Annotated, Optional
import logging
import traceback

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from .configuration import (
    Configuration,
    BaseOrderSchema,
    TrackingRequest,
    ModifyOrderQuantity,
    CancelOrderRequest,
    Product,
    CartItem,
    UserCart
)

logger = logging.getLogger('orders')


@tool
async def web_search(
        query: str,
        *,
        config: Annotated[RunnableConfig, InjectedToolArg]
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


class ProductDatabase:
    """Mock database for products"""

    def __init__(self):
        self.products = {
            "p1": Product("p1", "Smartphone X", Decimal("699.99"), "electronics",
                          "Latest smartphone with advanced features", 50),
            "p2": Product("p2", "Running Shoes", Decimal("89.99"), "sports",
                          "Comfortable running shoes for athletes", 100),
            "p3": Product("p3", "Coffee Maker", Decimal("129.99"), "appliances",
                          "Premium coffee maker with timer", 30)
        }
        self.carts: Dict[str, UserCart] = {}


db = ProductDatabase()

# Tool Implementations


@tool
async def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search for products in the store.

    Args:
        query: Search query string
        category: Optional category filter
        max_price: Optional maximum price filter
        min_price: Optional minimum price filter

    Returns:
        List of matching products with their details
    """
    results = []
    query = query.lower()

    for product in db.products.values():
        # Check if product matches search criteria
        matches_query = (
            query in product.name.lower() or
            query in product.description.lower()
        )
        matches_category = (
            not category or
            category.lower() == product.category.lower()
        )
        matches_price = (
            (not max_price or product.price <= Decimal(str(max_price))) and
            (not min_price or product.price >= Decimal(str(min_price)))
        )

        if matches_query and matches_category and matches_price:
            results.append({
                "id": product.id,
                "name": product.name,
                "price": float(product.price),
                "category": product.category,
                "description": product.description,
                "stock": product.stock
            })

    return results


@tool
async def get_cart(
    user_id: str,
) -> Dict[str, Any]:
    """
    Get the current shopping cart for a user.

    Args:
        user_id: The ID of the user whose cart to retrieve

    Returns:
        Cart details including items and total
    """
    cart = db.carts.get(user_id, UserCart(user_id, [], Decimal("0")))

    return {
        "user_id": cart.user_id,
        "items": [
            {
                "product_id": item.product_id,
                "product_name": db.products[item.product_id].name,
                "quantity": item.quantity,
                "price": float(item.price),
                "subtotal": float(item.price * item.quantity)
            }
            for item in cart.items
        ],
        "total": float(cart.total)
    }


@tool
async def update_cart(
    user_id: str,
    product_id: str,
    quantity: int
) -> Dict[str, Any]:
    """
    Update the shopping cart by adding/updating/removing products.

    Args:
        user_id: The ID of the user whose cart to update
        product_id: The ID of the product to update
        quantity: New quantity (0 to remove item)

    Returns:
        Updated cart details
    """
    # Get or create cart
    if user_id not in db.carts:
        db.carts[user_id] = UserCart(user_id, [], Decimal("0"))

    cart = db.carts[user_id]
    product = db.products.get(product_id)

    if not product:
        raise ValueError(f"Product {product_id} not found")

    if quantity > product.stock:
        raise ValueError(
            f"Requested quantity exceeds available stock ({product.stock})")

    # Find existing item
    existing_item = next(
        (item for item in cart.items if item.product_id == product_id),
        None
    )

    if quantity == 0 and existing_item:
        # Remove item
        cart.items.remove(existing_item)
    elif existing_item:
        # Update quantity
        existing_item.quantity = quantity
    elif quantity > 0:
        # Add new item
        cart.items.append(CartItem(product_id, quantity, product.price))

    # Recalculate total
    cart.total = sum(item.price * item.quantity for item in cart.items)

    return await get_cart(user_id)


# List of all available tools
TOOLS: List[Callable[..., Any]] = [
    web_search,
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
        ddg_results = await web_search(query, config={})
        print(f"Found {len(ddg_results)} results")

    asyncio.run(test_searches())
