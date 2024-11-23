# graph.py

"""Define a custom Reasoning and Order Support Agent using LangGraph.

Works with a chat model with tool calling support.

This module implements a stateful agent for handling order-related queries
using the official LangGraph implementation patterns.
"""
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Literal, TypedDict, Any, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .configuration import Configuration
from .prompt_manager import PromptManager
from .tool_manager import get_sensitive_tool_names, get_all_tools
from .tool_manager import TOOLS
from .state import InputState, ECommerseState
from .helper import load_chat_model

logger = logging.getLogger('orders')


# Define the function that calls the model


async def call_model(
    state: ECommerseState, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """

    try:
        configuration = Configuration.from_runnable_config(config)
        configurable = config.get("configurable", {})

        # Initialize the model with tool binding
        model = load_chat_model(configuration.model).bind_tools(TOOLS)

        # Get the system prompt template
        system_prompt = configurable.get("system_prompt")

        # Format the prompt with required parameters
        formatted_prompt = system_prompt.format_messages(
            user_info=configurable.get("user_info", "Guest User"),
            time=datetime.now(tz=timezone.utc).isoformat(),
            # Only use latest context
            messages=state.messages[-2:] if len(
                state.messages) > 2 else state.messages
        )

        # Get the model's response
        response = cast(
            AIMessage,
            await model.ainvoke(formatted_prompt, config)
        )

        # Handle last step case
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        content="Sorry, I could not find an answer to your question in the specified number of steps."
                    )
                ]
            }

        return {"messages": [response]}

    except Exception as e:
        logger.error(f"Error in call_model: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "messages": [
                AIMessage(
                    content="I apologize, but I encountered an error while processing your request. Please try again."
                )
            ]
        }

# Define the graph
builder = StateGraph(ECommerseState, input=InputState,
                     config_schema=Configuration)

# Add nodes
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint
builder.add_edge("__start__", "call_model")


def route_model_output(state: ECommerseState) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}")

    return "tools" if last_message.tool_calls else "__end__"


# Add edges
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
    {"tools": "tools", "__end__": "__end__"},
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Create memory saver and compile graph
memory = MemorySaver()
# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "Support Agent"
