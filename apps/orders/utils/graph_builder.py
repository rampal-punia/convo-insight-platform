from typing import Dict, List, Any, Optional
import logging
import os
import base64
from datetime import datetime
import traceback
from dataclasses import dataclass
from django.conf import settings

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from .prompt_manager import PromptManager
from .db_utils import DatabaseOperations
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
        self.db_ops = DatabaseOperations(user=config.order_details["user"])
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

        # Compiled graph reference
        self._compiled_graph = None

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

            # Check if state already has tracking info to prevent recursion
            tracking_info = state.get("tracking_info", {})

            # Prepare base context
            context = {
                "order_info": state["order_info"],
                "conversation_history": history,
                "user_input": user_input,
                "tracking_status": tracking_info.get("status", "Not available"),
                "shipping_method": tracking_info.get("shipping_method", "Not available"),
                "estimated_delivery": tracking_info.get("estimated_delivery", "Not available")
            }

            # If tracking info not in state and it's a tracking request, fetch it once
            if not tracking_info and self._is_tracking_request(user_input):
                try:
                    # Get tracking info from database only once
                    order_id = state["order_info"]["order_id"]
                    tracking_details = await self.db_ops.get_tracking_info(order_id)

                    # Store in state for future use
                    state["tracking_info"] = {
                        "status": tracking_details.get("status", "Not available"),
                        "shipping_method": tracking_details.get("shipping_method", "Not available"),
                        "estimated_delivery": tracking_details.get("estimated_delivery", "Not available"),
                        "current_location": tracking_details.get("current_location", "Not available"),
                        "last_update": tracking_details.get("latest_update", {})
                    }

                    # Update context with fetched info
                    context.update({
                        "tracking_status": state["tracking_info"]["status"],
                        "shipping_method": state["tracking_info"]["shipping_method"],
                        "estimated_delivery": state["tracking_info"]["estimated_delivery"]
                    })
                except Exception as e:
                    logger.error(f"Error fetching tracking info: {str(e)}")
                    # Continue with default values if fetch fails

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

    def _is_tracking_request(self, message_content: str) -> bool:
        """Enhanced method to check if the request is tracking-related"""
        tracking_keywords = {
            'track', 'where', 'delivery status', 'shipping status',
            'package location', 'delivery estimate', 'when will it arrive',
            'estimated delivery', 'delivery date', 'shipping date',
            'shipped', 'delivered', 'tracking', 'transit'
        }

        message_words = set(message_content.lower().split())
        return any(keyword in message_content.lower() for keyword in tracking_keywords) or \
            len(message_words.intersection(tracking_keywords)) > 0

    def _format_conversation_history(self, messages: List) -> str:
        """Format conversation history with message typing and max length"""
        if not messages:
            return ""

        # Only include last 5 messages to prevent context bloat
        recent_messages = messages[-5:]

        formatted_history = []
        for msg in recent_messages:
            role = 'User' if isinstance(msg, HumanMessage) else 'Assistant'
            # Truncate very long messages
            content = msg.content[:500] + \
                '...' if len(msg.content) > 500 else msg.content
            formatted_history.append(f"{role}: {content}")

        return "\n".join(formatted_history)

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
                    logger.info(
                        f"Calling tools from last message: {last_message}")
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and 'function' in tool_call:
                            tool_name = tool_call['function']['name']
                            logger.info(
                                f"Received tool name: {tool_name}")
                            if self.config.tool_manager.is_sensitive_tool(tool_name):
                                state["confirmation_pending"] = True
                                return END
                    logger.info(
                        f"Returning safe tool 'name': {tool_name}")
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
                # "agent": "agent",
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
        self._compiled_graph = workflow.compile(
            interrupt_before=["sensitive_tools"])
        return self._compiled_graph

    async def save_graph_visualization(self, output_path: Optional[str] = None) -> str:
        """Save the graph visualization as a PNG file asynchronously.

        Args:
            output_path (str, optional): Path where the PNG should be saved. 
                If None, saves to MEDIA_ROOT/graph_visualizations/

        Returns:
            str: Path to the saved visualization file
        """
        try:
            if not self._compiled_graph:
                raise ValueError("Graph must be built before visualization")

            # Get the mermaid PNG data using LangGraph's built-in method
            png_data = self._compiled_graph.get_graph().draw_mermaid_png()

            # Set up the output path
            if output_path is None:
                # Create directory if it doesn't exist
                vis_dir = os.path.join(
                    settings.MEDIA_ROOT, 'graph_visualizations')
                os.makedirs(vis_dir, exist_ok=True)

                # Generate unique filename using timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    vis_dir, f'graph_{self.config.intent}_{timestamp}.png')

            # Write the PNG data directly to file
            with open(output_path, 'wb') as f:
                f.write(png_data)

            logger.info(f"Graph visualization saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving graph visualization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_graph_base64(self) -> str:
        """Get the graph visualization as a base64 encoded string.

        Returns:
            str: Base64 encoded PNG data
        """
        try:
            if not self._compiled_graph:
                raise ValueError("Graph must be built before visualization")

            # Get the PNG data using LangGraph's built-in method
            png_data = self._compiled_graph.get_graph().draw_mermaid_png()

            # Convert to base64
            base64_data = base64.b64encode(png_data).decode('utf-8')
            return f"data:image/png;base64,{base64_data}"

        except Exception as e:
            logger.error(f"Error getting graph base64: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_graph_svg(self) -> str:
        """Get the graph visualization as SVG.

        Returns:
            str: SVG representation of the graph
        """
        try:
            if not self._compiled_graph:
                raise ValueError("Graph must be built before visualization")

            # Get the SVG directly from the graph
            return self._compiled_graph.get_graph().draw_mermaid_svg()

        except Exception as e:
            logger.error(f"Error getting graph SVG: {str(e)}")
            logger.error(traceback.format_exc())
            raise
