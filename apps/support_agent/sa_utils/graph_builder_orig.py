"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
import asyncio
from decouple import config

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from configuration import Configuration
from state import InputState, State
from tool_manager import TOOLS
from helper import load_chat_model

# Define the function that calls the model
logger = logging.getLogger('orders')


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
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
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith


async def run_search_query(query: str, max_steps: int = 7) -> List[Dict]:
    """
    Run a search query through the ReAct Agent graph.

    Args:
        query: The search query to process
        max_steps: Maximum number of steps before forcing completion

    Returns:
        List of response messages from the graph execution
    """
    # initialize configurations
    config = Configuration()

    # Create initial state with the search query
    state = InputState(
        messages=[
            HumanMessage(content=query)
        ]
    )

    # Set up config dict with recursion
    config_dict = {
        'recursion_limit': max_steps,
        'configurable': {
            "model": config.model,
            "max_search_results": config.max_search_results,
            "system_prompt": config.system_prompt
        }
    }

    try:
        result = await graph.ainvoke(input=state, config=config_dict)

        # Debug information
        print("\nDebug - Result type:", type(result))
        print("Debug - Result structure:", result)

        # The result is a dict with nested structure, extract messages from state
        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
        else:
            # Handle the case where result is the state object
            messages = result.messages if hasattr(result, 'messages') else []

        # Extract and return all AI messages
        ai_messages = [
            msg for msg in messages
            if isinstance(msg, AIMessage)
        ]
        return ai_messages

    except Exception as e:
        print(f"Error executing graph: {str(e)}")
        return []


async def main():
    """Main execution function."""
    TAVILY_API_KEY = config("TAVILY_API_KEY")

    # Get search query from user
    query = input("Enter your search query: ")

    print("\nProcessing query through ReAct Agent...\n")

    # Run the search
    messages = await run_search_query(query)

    # Print results
    print("\nSearch Results:")
    print("==============")

    for idx, msg in enumerate(messages, 1):
        print(f"\nResponse {idx}:")
        print("-------------")
        print(msg.content)

        # If there were tool calls, print those too
        if msg.tool_calls:
            print("\nTool Calls Used:")
            for tool_call in msg.tool_calls:
                print(
                    f"- Tool Name: {tool_call['name']} | tool args: {tool_call['args']}")
                # print(f"- {tool_call['name']}: {tool_call.function.name}")


if __name__ == '__main__':
    asyncio.run(main())
