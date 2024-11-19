from typing import Dict, List, Any
import logging
import traceback
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from .prompt_manager import PromptManager
from . import tool_manager as tm

logger = logging.getLogger('orders')


@dataclass
class GraphConfig:
    """Configuration class for graph builder"""
    llm: BaseChatModel
    intent: str
    order_details: dict
    tool_manager: tm.ToolManager
    conversation_id: str


class GraphBuilder:
    """Builds and configures LangGraph for order processing"""

    def __init__(self, config: GraphConfig):
        self.config = config
        self.prompt = PromptManager.initialize()
        self.prompt = PromptManager.get_prompt(self.config.intent)
        self.conversation_complete = False

        # Initialize tool sets
        self.safe_tools = self.config.tool_manager.get_safe_tools()
        self.sensitive_tools = self.config.tool_manager.get_sensitive_tools()

        # Bind tools to llm
        self.llm_with_safe_tools = self.config.llm.bind_tools(self.safe_tools)
        self.llm_with_sensitive_tools = self.config.llm.bind_tools(
            self.sensitive_tools)

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

            # For modify_order intent, check order status immediately
            if (state.get("intent") == "modify_order_quantity" and
                isinstance(latest_msg, HumanMessage) and
                "help" in latest_msg.content.lower() and
                    "modif" in latest_msg.content.lower()):

                order_status = state["order_info"]["status"]
                if order_status not in ["Pending", "Processing"]:
                    status_message = f"""I notice that your order is currently in '{order_status}' status. 
                    Unfortunately, orders can only be modified while they are in 'Pending' or 'Processing' status.
                    
                    Here's what you can do with your order in its current status:
                    """

                    status_actions = {
                        'Shipped': "- Track your shipment\n- Contact support for delivery updates",
                        'Delivered': "- Initiate a return (if within 30 days)\n- Leave a review",
                        'Cancelled': "- Place a new order\n- View similar products",
                        'Returned': "- Check refund status\n- Place a new order",
                        'In Transit': "- Track your shipment\n- Update delivery preferences"
                    }

                    status_message += "\n" + \
                        status_actions.get(
                            order_status, "- Contact customer support for assistance")

                    return {
                        "messages": [AIMessage(content=status_message)],
                        "completed": True
                    }

            # Format conversation history and get latest input
            history = self._format_conversation_history(state["messages"][:-1])
            user_input = latest_msg.content if isinstance(
                latest_msg, HumanMessage) else str(latest_msg)

            # Prepare context with tracking-specific info if needed
            context = {
                "order_info": state["order_info"],
                "conversation_history": history,
                "user_input": user_input,
                "tracking_status": "Not available",
                "shipping_method": "Not available",
                "estimated_delivery": "Not available"
            }

            if self._is_tracking_request(user_input):
                # Update tracking-specific context with actual values
                context.update({
                    "tracking_status": state["order_info"].get("status", "Not available"),
                    "shipping_method": state["order_info"].get("shipping_method", "Not available"),
                    "estimated_delivery": state["order_info"].get("estimated_delivery", "Not available")
                })

            # Generate response
            if self._requires_sensitive_tools(latest_msg):
                response = await self.sensitive_chain.ainvoke(context)
            else:
                response = await self.safe_chain.ainvoke(context)

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

    def _is_tracking_request(self, message_content: str) -> bool:
        """Check if the request is tracking-related"""
        tracking_keywords = [
            'track', 'where is', 'delivery status', 'shipping status',
            'package location', 'delivery estimate', 'when will it arrive'
        ]
        return any(keyword in message_content.lower() for keyword in tracking_keywords)

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
        logger.debug(f"Graph config: {self.config}")

        workflow = StateGraph(tm.OrderState)

        # Add nodes
        workflow.add_node("agent", self._agent_function)

        # Add separate nodes for safe and sensitive tools
        workflow.add_node("safe_tools", ToolNode(self.safe_tools))
        workflow.add_node("sensitive_tools", ToolNode(self.sensitive_tools))

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
