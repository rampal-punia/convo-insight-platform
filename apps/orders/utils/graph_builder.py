from typing import Dict, List, Any
import logging
import traceback
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from .prompt_manager import PromptManager
from . import tool_manager as tm

logger = logging.getLogger('orders')  # Get the orders logger


@dataclass
class GraphConfig:
    """Configuration class for graph builder"""
    llm: BaseChatModel
    intent: str
    order_details: dict
    tool_manager: tm.ToolManager  # ToolManager instance
    conversation_id: str


class GraphBuilder:
    """Builds and configures LangGraph for order processing"""

    def __init__(self, config: GraphConfig):
        self.config = config
        self.prompt = PromptManager.initialize()
        self.prompt = PromptManager.get_prompt(self.config.intent)
        self.conversation_complete = False

        # Initialize tool nodes
        safe_tools = self.config.tool_manager.get_safe_tools()
        sensitive_tools = self.config.tool_manager.get_sensitive_tools()

        # Bind tools to llm
        self.llm_with_safe_tools = self.config.llm.bind_tools(safe_tools)
        self.llm_with_sensitive_tools = self.config.llm.bind_tools(
            sensitive_tools)

        # Create chains
        self.safe_chain = self.prompt | self.llm_with_safe_tools
        self.sensitive_chain = self.prompt | self.llm_with_sensitive_tools

        # Message deduplication
        self.processed_messages = set()

    async def _agent_function(self, state: Dict) -> Dict:
        """Agent function that processes the state and generates responses"""
        try:
            # Check completion state
            if state.get("completed", False):
                return {"messages": []}

            # Get the latest message
            latest_msg = state["messages"][-1] if state["messages"] else None
            if not latest_msg:
                return {"messages": []}

            # Skip if the last message was from the AI
            if isinstance(latest_msg, AIMessage):
                return {"messages": []}

            # Format conversation history and get latest input
            history = self._format_conversation_history(state["messages"][:-1])
            user_input = latest_msg.content if isinstance(
                latest_msg, HumanMessage) else str(latest_msg)

            # Prepare input for chains
            chain_input = {
                "order_info": state["order_info"],
                "conversation_history": history,
                "user_input": user_input,
            }

            # Generate response
            if self._requires_sensitive_tools(latest_msg):
                response = await self.sensitive_chain.ainvoke(chain_input)
            else:
                response = await self.safe_chain.ainvoke(chain_input)

            # Update state
            if self._is_conversation_complete(response):
                state["completed"] = True

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error in agent function: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "messages": [],
                "error": str(e)
            }

    def _is_conversation_complete(self, response: AIMessage) -> bool:
        """Determines if the conversation is complete"""
        try:
            if not response:
                return False

            # Check for tool calls in additional_kwargs
            if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                tool_calls = response.additional_kwargs['tool_calls']
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and 'function' in tool_call:
                        tool_name = tool_call['function']['name']
                        if self.config.tool_manager.is_sensitive_tool(tool_name):
                            return True

            # If there's no content, return False
            if not hasattr(response, 'content') or not response.content:
                return False

            content = response.content.lower()
            # Check for completion indicators in the content
            completion_indicators = [
                "is there anything else",
                "can i help you with anything else",
                "let me know if you need anything else",
                "your order status is",
                "would you like to proceed",
            ]
            return any(indicator in content for indicator in completion_indicators)

        except Exception as e:
            logger.error(f"Error in _is_conversation_complete: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _requires_sensitive_tools(self, message: Any) -> bool:
        """Determine if the message requires sensitive tools"""
        if not hasattr(message, 'content'):
            return False

        content = message.content.lower()
        sensitive_keywords = [
            'modify', 'change', 'cancel', 'update', 'delete',
            'remove', 'refund', 'return'
        ]
        return any(keyword in content for keyword in sensitive_keywords)

    def _get_next_node(self, state: Dict) -> str:
        """Determines the next node in the graph"""
        try:
            # Check for completion states
            if state.get("completed", False) or self.conversation_complete:
                return END

            if state.get("confirmation_pending", False):
                return END

            # Get last message
            last_message = state["messages"][-1] if state["messages"] else None
            if not last_message:
                return END

            # Check for tool calls in additional_kwargs
            if isinstance(last_message, AIMessage):
                if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
                    tool_calls = last_message.additional_kwargs['tool_calls']
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and 'function' in tool_call:
                            tool_name = tool_call['function']['name']
                            if self.config.tool_manager.is_sensitive_tool(tool_name):
                                state["confirmation_pending"] = True
                                return END
                    return "safe_tools"

                # Check for completion
                if self._is_conversation_complete(last_message):
                    state["completed"] = True
                    return END

            # Check for duplicate messages
            msg_id = hash(f"{last_message.content}_{len(state['messages'])}" if hasattr(
                last_message, 'content') else str(last_message))

            if not hasattr(self, 'processed_messages'):
                self.processed_messages = set()

            if msg_id in self.processed_messages:
                return END

            self.processed_messages.add(msg_id)

            return "agent"

        except Exception as e:
            logger.error(f"Error in _get_next_node: {str(e)}")
            logger.error(traceback.format_exc())
            return "agent"

    async def _should_continue(self, state: Dict) -> bool:
        """Determines if the conversation should continue"""
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
                "To assist you with modifying order #"
            ]
            if any(indicator in content for indicator in completion_indicators):
                self.conversation_complete = True
                return False

        return True

    def _format_conversation_history(self, messages: List) -> str:
        """Formats the conversation history for the prompt"""
        return "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages[:-1]  # Exclude the last message
        ])

    def build(self) -> StateGraph:
        """Builds and returns the configured StateGraph with separate tool handling"""
        logger.info(f"Building graph for intent: {self.config.intent}")

        workflow = StateGraph(tm.OrderState)

        # Add nodes
        workflow.add_node("agent", self._agent_function)

        # Add separate nodes for safe and sensitive tools
        workflow.add_node("safe_tools",
                          ToolNode(self.config.tool_manager.get_safe_tools()))
        workflow.add_node("sensitive_tools",
                          ToolNode(self.config.tool_manager.get_sensitive_tools()))

        # Add edges
        workflow.add_edge(START, "agent")

        # Add conditional edges with enhanced routing
        workflow.add_conditional_edges(
            "agent",
            self._get_next_node,
            {
                "agent": "agent",
                "safe_tools": "safe_tools",
                "sensitive_tools": "sensitive_tools",
                END: END
            }
        )

        # Route from tool nodes back to agent
        workflow.add_edge("safe_tools", "agent")
        workflow.add_edge("sensitive_tools", "agent")

        logger.debug("Graph building completed, compiling workflow")

        # Only interrupt before sensitive tools
        return workflow.compile(interrupt_before=["sensitive_tools"])
