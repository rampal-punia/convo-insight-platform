from typing import Dict, List, Any
import logging
import traceback
from dataclasses import dataclass
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from . import tool_manager as tm

logger = logging.getLogger('orders')  # Get the orders logger


class PromptManager:
    """Manages prompts for different intents"""

    @staticmethod
    def get_prompt(intent: str) -> str:
        prompts = {
            "modify_order": """You are an order modification specialist. You help customers by:
                            1. First explaining what you're going to do
                            2. Using tools to check modification options
                            3. Providing clear guidance highlighting on possible changes

                            Current order details:
                            {order_info}

                            Previous conversation context:
                            {conversation_history}

                            New request: {user_input}

                            If a tool has been used, always provide a clear confirmation or explanation of what happened.""",

            "order_status": """You are an order status assistant. Help customers by checking order status and providing updates.
                            Current order details: {order_info}

                            Previous conversation context:
                            {conversation_history}

                            New request: {user_input}

                            If a tool has been used, always provide a clear confirmation or explanation of what happened.""",

            "return_request": """You are a returns specialist. You help customers by:
                            1. First explaining what you're going to do
                            2. Using tools to check eligibility and process returns
                            3. Providing clear, step-by-step instructions

                            Current order details: {order_info}

                            Previous conversation context:
                            {conversation_history}

                            New request: {user_input}

                            If a tool has been used, always provide a clear confirmation or explanation of what happened.""",

            "delivery_issue": """You are a delivery support specialist. You help customers by:
                            1. First explaining what you're going to do
                            2. Using tools to check delivery status
                            3. Providing solutions and next steps

                            Current order details: {order_info}

                            Previous conversation context:
                            {conversation_history}

                            New request: {user_input}
                            If a tool has been used, always provide a clear confirmation or explanation of what happened.""",
        }
        return prompts.get(intent, prompts["modify_order"])


@dataclass
class GraphConfig:
    """Configuration class for graph builder"""
    llm: BaseChatModel
    intent: str
    order_details: dict
    tools: list
    conversation_id: str


class GraphBuilder:
    """Builds and configures LangGraph for order processing"""

    def __init__(self, config: GraphConfig):
        self.config = config
        self.prompt = self._create_prompt()
        self.llm_with_tools = self.config.llm.bind_tools(self.config.tools)
        self.chain = self.prompt | self.llm_with_tools
        self.conversation_complete = False

    def _create_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt template based on intent"""
        prompt_text = PromptManager.get_prompt(self.config.intent)
        return ChatPromptTemplate.from_messages([
            ("system", prompt_text),
        ])

    async def _agent_function(self, state: tm.OrderState) -> Dict:
        """Agent function that processes the state and generates responses"""
        try:
            logger.debug(f"Agent processing state: {state}")

            # Handle tool message if present
            last_message = state["messages"][-1] if state["messages"] else None

            # Don't process if we're already complete
            if self.conversation_complete:
                return {"messages": state["messages"]}

            if isinstance(last_message, ToolMessage):
                logger.info("Processing tool response")
                tool_response = f"Previous action result: {last_message.content}"
                state["messages"].append(HumanMessage(content=tool_response))

            # Format history and get input
            history = self._format_conversation_history(state["messages"][:-1])
            user_input = last_message.content if hasattr(
                last_message, "content") else str(last_message)

            # Invoke chain with context
            response = await self.chain.ainvoke({
                "order_info": state["order_info"],
                "conversation_history": history,
                "user_input": user_input,
            })

            logger.info(f"LLM Response: {response}")

            # Check if this should complete the conversation
            if isinstance(response, AIMessage):
                content = response.content.lower() if response.content else ""
                if any(indicator in content for indicator in [
                    "your order status is",
                    "current status of your order is",
                    "tracking information shows"
                ]):
                    self.conversation_complete = True

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error in agent function: {str(e)}")
            return {
                "messages": [
                    AIMessage(
                        content="I apologize, but I encountered an error. Please try again."
                    )
                ]
            }

    def _should_continue(self, state: tm.OrderState) -> bool:
        """Determines if the conversation should continue based on state"""
        # Check if there are any pending tool operations
        if self.conversation_complete:
            return False

        if state.get("confirmation_pending"):
            return True

        # Check if the last message indicates a completion
        last_message = state["messages"][-1] if state["messages"] else None
        if isinstance(last_message, AIMessage):
            content = last_message.content.lower() if last_message.content else ""
            completion_indicators = [
                "is there anything else",
                "can i help you with anything else",
                "is there something else",
                "let me know if you need anything else",
                "your order status is",  # Specific to order status checks
                "current status of your order is",
                "tracking information shows",
            ]
            if any(indicator in content for indicator in completion_indicators):
                self.conversation_complete = True
                return False

        return True

    def _tool_router(self, state: tm.OrderState) -> str:
        """Routes to appropriate node based on state and tool calls"""
        # Check for tool calls in the last message
        last_message = state["messages"][-1] if state["messages"] else None

        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            # Specific tool routing logic
            tool_name = last_message.tool_calls[0].get('name')
            if tool_name in ["modify_order_quantity", "cancel_order"]:
                state["confirmation_pending"] = True
            return "tools"

        # Check if we just completed a tool execution
        if state.get("confirmation_pending"):
            state["confirmation_pending"] = False
            return "agent"

        # Check completion status
        if self.conversation_complete or not self._should_continue(state):
            return END

        # Only return to agent if we have a valid reason
        if isinstance(last_message, (HumanMessage, ToolMessage)):
            return "agent"

        return END

    def _format_conversation_history(self, messages: List) -> str:
        """Formats the conversation history for the prompt"""
        return "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            # Exclude the last message as it's the current input
            for msg in messages[:-1]
        ])

    def build(self) -> StateGraph:
        """Builds and returns the configured StateGraph with enhanced flow control"""
        # Log the initialization of graph building
        logger.info(f"Building graph for intent: {self.config.intent}")

        workflow = StateGraph(tm.OrderState)

        # Add nodes
        workflow.add_node("agent", self._agent_function)
        workflow.add_node("tools", ToolNode(self.config.tools))

        # Add edges with enhanced routing
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._tool_router,
            {
                "agent": "agent",
                "tools": "tools",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "tools",
            self._tool_router,
            {
                "agent": "agent",
                "tools": "tools",
                END: END
            }
        )

        # Log completion of graph building
        logger.debug("Graph building completed, compiling workflow")

        # Compile with only supported parameters
        return workflow.compile(interrupt_before=["tools"])
