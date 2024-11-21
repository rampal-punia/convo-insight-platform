"""Define the Order Support Agent using LangGraph.

This module implements a stateful agent for handling order-related queries
using the official LangGraph implementation patterns.
"""
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Literal, TypedDict, cast
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated

from .configuration import Configuration
from .prompt_manager import PromptManager
from .tool_manager import TOOLS, get_sensitive_tool_names
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

            # Get appropriate prompt template
            prompt = PromptManager.get_prompt(intent)

            # Format the system prompt with context
            formatted_prompt = prompt.format(
                order_info=order_info,
                conversation_history=self._format_history(
                    state['messages'][:-1]),
                user_input=state['messages'][-1].content,
                system_time=datetime.now(tz=timezone.utc).isoformat()
            )

            # Get model response
            while True:
                result = await self.runnable.ainvoke(
                    [{"role": "system", "content": formatted_prompt},
                        *state['messages']]
                )

                # Validate response
                if not result.tool_calls and (not result.content or
                                              isinstance(result.content, list) and not result.content[0].get("text")):
                    state['messages'].append(HumanMessage(
                        content="Please provide a clear response."))
                    continue
                break

            return {"messages": [result]}

        except Exception as e:
            logger.error(f"Error in Assistant call: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "messages": [
                    AIMessage(
                        content="I encountered an error processing your request. Please try again.")
                ]
            }

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


def create_customer_support_agent(config: Configuration):
    """Create and configure the order support agent."""

    # Initialize LLM with tools
    model = load_chat_model(config['llm']).bind_tools(TOOLS)
    sensitive_tools = get_sensitive_tool_names()

    # Create assistant with prompt template
    assistant = Assistant(model)

    # Initialize graph
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("safe_tools", ToolNode([t for t in TOOLS
                                             if t.name not in sensitive_tools]))
    builder.add_node("sensitive_tools", ToolNode([t for t in TOOLS
                                                  if t.name in sensitive_tools]))

    # Add edges
    builder.add_edge("__start__", "assistant")

    # Route based on tool sensitivity
    def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
        """Route to appropriate tool node based on tool sensitivity."""
        next_node = tools_condition(state)
        if next_node == "__end__":
            return "__end__"

        ai_message = state["messages"][-1]
        if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
            return "__end__"

        tool_name = ai_message.tool_calls[0]["name"]
        return "sensitive_tools" if tool_name in get_sensitive_tool_names else "safe_tools"

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

    # Add return edges to assistant
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("sensitive_tools", "assistant")

    # Compile graph with interrupts for sensitive tools
    return builder.compile(interrupt_before=["sensitive_tools"])


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
