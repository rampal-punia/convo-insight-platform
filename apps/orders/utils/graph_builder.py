from typing import Dict, List, Any
import logging
import traceback
from dataclasses import dataclass

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
                            3. Providing clear guidance on possible changes

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
            if isinstance(last_message, ToolMessage):
                logger.info("Processing tool response")
                tool_response = f"Previous action result: {last_message.content}"
                state["messages"].append(HumanMessage(content=tool_response))

            # Format conversation history
            messages = state["messages"]
            history = self._format_conversation_history(messages)

            # Get latest input
            latest_message = messages[-1]
            user_input = latest_message.content if hasattr(
                latest_message, "content") else str(latest_message)

            # Invoke chain with context
            response = await self.chain.ainvoke({
                "order_info": state["order_info"],
                "conversation_history": history,
                "user_input": user_input,
            })

            logger.info(f"LLM Response: {response}")

            # Handle tool calls
            if not response.content and response.tool_calls:
                return {
                    "messages": [
                        AIMessage(
                            content="Let me help you with that modification...",
                            tool_calls=response.tool_calls
                        )
                    ]
                }
            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error in agent function: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "messages": [
                    AIMessage(
                        content="I apologize, but I encountered an error processing your request. Could you please try rephrasing your question?"
                    )
                ]
            }

    def _format_conversation_history(self, messages: List) -> str:
        """Formats the conversation history for the prompt"""
        return "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            # Exclude the last message as it's the current input
            for msg in messages[:-1]
        ])

    def build(self) -> StateGraph:
        """Builds and returns the configured StateGraph"""
        builder = StateGraph(tm.OrderState)

        # Add nodes
        builder.add_node("agent", self._agent_function)
        builder.add_node('tools', ToolNode(self.config.tools))

        # Add edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            'agent',
            tools_condition,
        )
        builder.add_edge("tools", "agent")
        return builder.compile(interrupt_before=["tools"])
