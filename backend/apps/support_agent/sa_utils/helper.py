"""Utility & helper functions."""

from typing import Any, Callable, List, cast, Dict, Annotated
from datetime import datetime, timezone
import logging
import traceback

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger('orders')


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (
            c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def format_chat_history(messages: List[AnyMessage], max_messages: int = 5) -> str:
    """Format the conversation history."""
    if not messages:
        return ""

    recent = messages[-max_messages:]
    formatted = []

    for msg in recent:
        role = 'User' if isinstance(msg, HumanMessage) else 'Assistant'
        content = msg.content[:500] + \
            '...' if len(msg.content) > 500 else msg.content
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def format_order_details(order_info: Dict) -> str:
    """Format order details into a readable message."""
    try:
        items_str = "\n".join([
            f"• {item['product_name']} (Quantity: {item['quantity']}, Price: ${item['price']})"
            for item in order_info.get('items', [])
        ])

        return f"""
Here are the details for Order #{order_info.get('order_id')}:

Status: {order_info.get('status')}

Items:
{items_str}

Total Amount: ${order_info.get('total_amount')}

You can ask me to:
• Track this order
• Modify quantities (if order is pending/processing)
• Cancel the order (if eligible)
"""
    except Exception as e:
        logger.error(f"Error formatting order details: {str(e)}")
        logger.error(traceback.format_exc())
        return "I'm having trouble formatting the order details. Please try again."
