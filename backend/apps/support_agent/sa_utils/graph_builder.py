# graph.py

"""LangGraph v2 Support Agent for e-commerce order management.

Implements an agent loop: call_model -> route (tools or end) -> tools -> call_model.
Uses MemorySaver checkpointer for conversation persistence across turns.
"""
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .configuration import Configuration
from .tool_manager import TOOLS
from .state import InputState, ECommerceState
from .helper import load_chat_model

logger = logging.getLogger('orders')


async def call_model(
    state: ECommerceState, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM with the conversation history and bound tools.

    The model decides whether to respond directly or invoke a tool.
    """
    try:
        configuration = Configuration.from_runnable_config(config)
        configurable = config.get("configurable", {})

        # Bind tools so the model can decide to call them
        model = load_chat_model(configuration.model).bind_tools(TOOLS)

        # Get and format the system prompt with customer context
        system_prompt = configurable.get("system_prompt")
        if system_prompt is None:
            raise ValueError("system_prompt not provided in config['configurable']")

        formatted_messages = system_prompt.format_messages(
            customer_id=str(configurable.get("customer_id", "")),
            username=configurable.get("username", "Customer"),
            time=datetime.now(tz=timezone.utc).isoformat(),
            messages=state.messages,
        )

        response = cast(
            AIMessage,
            await model.ainvoke(formatted_messages, config),
        )

        # Safety: if we're on the last step and the model still wants tools,
        # force a text-only response instead of looping further.
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        content="Sorry, I wasn't able to complete that request. "
                                "Could you try rephrasing or start a new query?"
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
                    content="I apologize, but I encountered an error while "
                            "processing your request. Please try again."
                )
            ]
        }


def route_model_output(state: ECommerceState) -> Literal["tools", END]:
    """Route to 'tools' if the model wants to call them, otherwise END."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage, got {type(last_message).__name__}"
        )
    return "tools" if last_message.tool_calls else END


# ── Build the graph ──────────────────────────────────────────────────────────

builder = StateGraph(
    ECommerceState,
    input=InputState,
    config_schema=Configuration,
)

# Nodes
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Edges
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    route_model_output,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "call_model")

# Compile with checkpointer for multi-turn memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
graph.name = "Support Agent"
