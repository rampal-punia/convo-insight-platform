"""Define the Order Support Agent using LangGraph.

This module implements a stateful agent for handling order-related queries
using the official LangGraph implementation patterns.
"""
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Literal, TypedDict, Any
import json

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated
from .context_manager import CustomJSONEncoder
from .configuration import Configuration
from .prompt_manager import PromptManager
from .tool_manager import get_sensitive_tool_names, get_all_tools
from .helper import load_chat_model

logger = logging.getLogger('orders')


class State(TypedDict):
    """Represents the state of the conversation."""
    messages: Annotated[List[AnyMessage], add_messages]
    order_info: Dict
    intent: str
    conversation_id: str
    confirmation_pending: bool
    completed: bool
    context: Dict[str, Any]  # Add context field to maintain additional state


class Assistant:
    """Assistant class that manages the LLM interactions."""

    def __init__(self, runnable):
        self.runnable = runnable

    async def __call__(self, state: State, config: RunnableConfig) -> Dict:
        """Process the current state and generate a response."""
        try:
            # Get conversation context
            intent = state.get('intent', 'general')
            logger.info(f"Intent is {intent}")
            logger.info(f"state is {state}")
            order_info = state.get('order_info', {})
            context = state.get('context', {})

            # For order_detail intent, format response directly
            if intent == 'order_detail' and order_info:
                return {
                    "messages": [
                        AIMessage(
                            content=self._format_order_details(order_info))
                    ]
                }

            # Get appropriate prompt template
            prompt = PromptManager.get_prompt(intent)
            logger.info(f"Generated prompt: {prompt}")

            # Format the system prompt with context
            formatted_prompt = prompt.format(
                order_info=json.dumps(
                    order_info, indent=2, cls=CustomJSONEncoder),
                conversation_history=self._format_history(
                    state['messages'][:-1]),
                user_input=state['messages'][-1].content,
                system_time=datetime.now(tz=timezone.utc).isoformat(),
                context=json.dumps(context, indent=2)
            )

            # Get model response
            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "system",
                    "content": f"Current order context: {json.dumps(order_info, indent=2, cls=CustomJSONEncoder)}"},
                *state['messages']
            ]

            # Get model response with validation loop
            while True:
                result = await self.runnable.ainvoke(messages)

                # Validate response
                if not result.tool_calls and (
                    not result.content or
                    isinstance(result.content,
                               list) and not result.content[0].get("text")
                ):
                    logger.warning("Invalid response from LLM, retrying...")
                    messages.append(HumanMessage(
                        content="Please provide a clear response with proper context awareness."
                    ))
                    continue

                # Additional validation for context preservation
                if order_info and not self._validates_order_context(result):
                    logger.warning(
                        "Response missing order context, reinforcing...")
                    messages.append(HumanMessage(
                        content="Please ensure your response acknowledges the current order context."
                    ))
                    continue

                break

            return {"messages": [result]}

        except Exception as e:
            logger.error(f"Error in Assistant call: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "messages": [
                    AIMessage(
                        content="I encountered an error processing your request. Please try again."
                    )
                ]
            }

    def _validates_order_context(self, result: AIMessage) -> bool:
        """
        Validate that the response maintains awareness of order context.
        This is a basic check that can be enhanced based on your specific needs.
        """
        if not hasattr(result, 'content'):
            return False

        content = result.content.lower()

        # Check for common order-related terms
        context_indicators = [
            'order', 'item', 'purchase', 'delivery',
            'tracking', 'status', 'shipping'
        ]

        # If using tool calls, those are also valid responses
        if hasattr(result, 'tool_calls') and result.tool_calls:
            return True

        return any(indicator in content for indicator in context_indicators)

    def _format_order_details(self, order_info: Dict) -> str:
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
            return "I'm having trouble formatting the order details. Please try again."

    def _format_history(self, messages: List[AnyMessage], max_messages: int = 5) -> str:
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


def create_customer_support_agent(config: Dict[str, Any]):
    """Create and configure the order support agent."""
    try:
        # Initialize LLM with tools
        model = load_chat_model(config['llm'])
        logger.info(f"Loaded chat model: {config['llm']}")

        # Log available tools
        tools = get_all_tools()
        logger.info(f"Available tools: {[t.name for t in tools]}")

        # Bind tools to model
        bound_model = model.bind_tools(tools)
        logger.info(f"Bound {len(tools)} tools to model")

        # Create assistant
        assistant = Assistant(bound_model)

        # Initialize graph
        builder = StateGraph(State)

        # Add nodes
        builder.add_node("assistant", assistant)

        safe_tools = [
            t for t in tools if t.name not in get_sensitive_tool_names()]
        sensitive_tools = [
            t for t in tools if t.name in get_sensitive_tool_names()]

        builder.add_node("safe_tools", ToolNode(safe_tools))
        builder.add_node("sensitive_tools", ToolNode(sensitive_tools))

        logger.info(
            f"Created nodes with {len(safe_tools)} safe tools and {len(sensitive_tools)} sensitive tools")

        # Add basic edge from start to assistant
        builder.add_edge("__start__", "assistant")

        def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
            """Route to appropriate tool node or end based on intent and message"""
            logger.info(f"Routing with intent: {state.get('intent')}")

            # For order_detail intent, we don't need tools
            if state.get('intent') == 'order_detail':
                logger.info("Order detail intent - no tools needed")
                return "__end__"

            # For other intents, check for tool calls
            next_node = tools_condition(state)
            if next_node == "__end__":
                return "__end__"

            ai_message = state["messages"][-1]
            if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
                logger.info("No tool calls in message, ending")
                return "__end__"

            tool_name = ai_message.tool_calls[0]["name"]
            logger.info(
                f"Routing to {'sensitive_tools' if tool_name in get_sensitive_tool_names() else 'safe_tools'} for tool: {tool_name}")
            return "sensitive_tools" if tool_name in get_sensitive_tool_names() else "safe_tools"

        # Add conditional edges
        builder.add_conditional_edges(
            "assistant",
            route_tools,
            {
                "safe_tools": "safe_tools",
                "sensitive_tools": "sensitive_tools",
                "__end__": "__end__"
            }
        )

        # Add return edges
        builder.add_edge("safe_tools", "assistant")
        builder.add_edge("sensitive_tools", "assistant")

        logger.info("Graph creation complete")
        return builder.compile(interrupt_before=["sensitive_tools"])

    except Exception as e:
        logger.error(f"Error creating support agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise


async def run_order_support(
    query: str,
    order_info: Dict,
    intent: str,
    conversation_id: str,
    max_steps: int = 7
) -> List[Dict]:
    """Run a query through the Order Support Agent.

    Args:
        query: User's query
        order_info: Order context information
        intent: Conversation intent
        conversation_id: Unique conversation identifier
        max_steps: Maximum number of steps before completion

    Returns:
        List of response messages
    """
    try:
        # Initialize configuration
        config = Configuration()

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "order_info": order_info,
            "intent": intent,
            "conversation_id": conversation_id,
            "confirmation_pending": False,
            "completed": False
        }

        # Configure graph settings
        config_dict = {
            "recursion_limit": max_steps,
            "configurable": {
                "model": config.model,
                "system_prompt": config.system_prompt,
                "max_retries": config.max_tool_retries,
                "tool_timeout": config.tool_timeout
            }
        }

        # Create and run graph
        graph = create_customer_support_agent(config)
        result = await graph.ainvoke(initial_state, config_dict)

        # Extract messages
        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
        else:
            messages = result.messages if hasattr(result, 'messages') else []

        return [msg for msg in messages if isinstance(msg, AIMessage)]

    except Exception as e:
        logger.error(f"Error running order support: {str(e)}")
        logger.error(traceback.format_exc())
        return []
