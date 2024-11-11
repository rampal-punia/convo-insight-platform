```python
from typing import Annotated, Literal, Dict, List
from typing_extensions import TypedDict
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# State Definition
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order_info: dict
    intent: str
    conversation_id: str
    modified: bool  # Track if order was modified
    confirmation_pending: bool  # Track if waiting for confirmation

# Tool Definitions
class ModifyOrderRequest(BaseModel):
    order_id: int = Field(description="The ID of the order to modify")
    new_details: dict = Field(description="The new details for the order")
    confirmation: bool = Field(description="Whether this is a confirmation of changes")

# LLM Setup
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Prompts
order_modification_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a specialized order modification assistant for an e-commerce platform. "
     "Help customers modify their orders while adhering to company policies. "
     "Current order details:\n{order_info}\n"
     "Intent: {intent}\n"
     "If the user confirms changes, use ModifyOrderRequest with confirmation=True."
     "If changes are rejected, apologize and ask if they'd like to try different modifications."
    ),
    ("human", "{input}")
])

def create_order_support_graph():
    builder = StateGraph(OrderState)
    
    # Initialize State
    def initialize_state(state: OrderState):
        return {
            "order_info": get_order_details(state.get("order_id")),
            "modified": False,
            "confirmation_pending": False
        }
    
    # Order Modification Node
    def order_modifier(state: OrderState):
        messages = state["messages"]
        order_info = state["order_info"]
        
        if state.get("confirmation_pending"):
            last_message = messages[-1]
            if "confirm" in last_message.content.lower():
                return {
                    "messages": [AIMessage(content="Processing your confirmation...")],
                    "modified": True,
                    "tool_calls": [{"name": "ModifyOrderRequest", "confirmation": True}]
                }
            else:
                return {
                    "messages": [AIMessage(content="Changes cancelled. Would you like to try different modifications?")],
                    "confirmation_pending": False
                }
        
        response = llm.invoke(
            order_modification_prompt.format(
                order_info=order_info,
                intent="modify_order",
                input=messages[-1].content
            )
        )
        
        return {
            "messages": [response],
            "confirmation_pending": True if "confirm" in response.content.lower() else False
        }
    
    # Tool Handling Node
    def tool_handler(state: OrderState):
        if state.get("modified"):
            # Process actual order modification in database
            update_order_in_db(state["order_info"])
            return {
                "messages": [AIMessage(content="Order successfully modified! Is there anything else you need help with?")]
            }
        return {
            "messages": state["messages"]
        }
    
    # Add Nodes
    builder.add_node("initialize", initialize_state)
    builder.add_node("modifier", order_modifier)
    builder.add_node("tools", tool_handler)
    
    # Add Edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "modifier")
    
    def route_next(state: OrderState):
        if state.get("modified"):
            return "tools"
        if state.get("confirmation_pending"):
            return "modifier"
        return END
    
    builder.add_conditional_edges(
        "modifier",
        route_next,
        {
            "tools": "tools",
            "modifier": "modifier",
            END: END
        }
    )
    
    builder.add_edge("tools", END)
    
    return builder.compile()

# Helper Functions
def get_order_details(order_id: int) -> dict:
    """Fetch order details from database"""
    from .models import Order
    order = Order.objects.get(id=order_id)
    return {
        "id": order.id,
        "status": order.status,
        "items": [
            {
                "id": item.id,
                "product": item.product.name,
                "quantity": item.quantity,
                "price": float(item.price)
            }
            for item in order.items.all()
        ],
        "total_amount": float(order.total_amount)
    }

def update_order_in_db(order_info: dict) -> None:
    """Update order in database"""
    from .models import Order, OrderItem
    order = Order.objects.get(id=order_info["id"])
    
    for item_info in order_info["items"]:
        item = order.items.get(id=item_info["id"])
        item.quantity = item_info["quantity"]
        item.price = item_info["price"]
        item.save()
    
    order.total_amount = sum(item.price for item in order.items.all())
    order.save()

```

## Async graph
```python
import asyncio
from functools import partial
from typing import Annotated, Literal, Dict, List
from typing_extensions import TypedDict
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from channels.db import database_sync_to_async

# State Definition
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order_info: dict
    intent: str
    conversation_id: str
    modified: bool
    confirmation_pending: bool

class ModifyOrderRequest(BaseModel):
    order_id: int = Field(description="The ID of the order to modify")
    new_details: dict = Field(description="The new details for the order")
    confirmation: bool = Field(description="Whether this is a confirmation of changes")

# Async Database Operations
@database_sync_to_async
def get_order_details_async(order_id: int) -> dict:
    """Async wrapper for fetching order details"""
    from .models import Order
    order = Order.objects.get(id=order_id)
    return {
        "id": order.id,
        "status": order.status,
        "items": [
            {
                "id": item.id,
                "product": item.product.name,
                "quantity": item.quantity,
                "price": float(item.price)
            }
            for item in order.items.all()
        ],
        "total_amount": float(order.total_amount)
    }

@database_sync_to_async
def update_order_in_db_async(order_info: dict) -> None:
    """Async wrapper for updating order in database"""
    from .models import Order, OrderItem
    order = Order.objects.get(id=order_info["id"])
    
    for item_info in order_info["items"]:
        item = order.items.get(id=item_info["id"])
        item.quantity = item_info["quantity"]
        item.price = item_info["price"]
        item.save()
    
    order.total_amount = sum(item.price for item in order.items.all())
    order.save()

# LLM Setup with Async Support
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Prompts
order_modification_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a specialized order modification assistant for an e-commerce platform. "
     "Help customers modify their orders while adhering to company policies. "
     "Current order details:\n{order_info}\n"
     "Intent: {intent}\n"
     "If the user confirms changes, use ModifyOrderRequest with confirmation=True."
     "If changes are rejected, apologize and ask if they'd like to try different modifications."
    ),
    ("human", "{input}")
])

async def create_order_support_graph():
    """Async factory function for creating the graph"""
    builder = StateGraph(OrderState)
    
    # Initialize State
    async def initialize_state(state: OrderState):
        order_info = await get_order_details_async(state.get("order_id"))
        return {
            "order_info": order_info,
            "modified": False,
            "confirmation_pending": False
        }
    
    # Order Modification Node
    async def order_modifier(state: OrderState):
        messages = state["messages"]
        order_info = state["order_info"]
        
        if state.get("confirmation_pending"):
            last_message = messages[-1]
            if "confirm" in last_message.content.lower():
                return {
                    "messages": [AIMessage(content="Processing your confirmation...")],
                    "modified": True,
                    "tool_calls": [{"name": "ModifyOrderRequest", "confirmation": True}]
                }
            else:
                return {
                    "messages": [AIMessage(content="Changes cancelled. Would you like to try different modifications?")],
                    "confirmation_pending": False
                }
        
        # Use asyncio.to_thread for CPU-bound LLM operations
        response = await asyncio.to_thread(
            llm.invoke,
            order_modification_prompt.format(
                order_info=order_info,
                intent="modify_order",
                input=messages[-1].content
            )
        )
        
        return {
            "messages": [response],
            "confirmation_pending": True if "confirm" in response.content.lower() else False
        }
    
    # Tool Handling Node
    async def tool_handler(state: OrderState):
        if state.get("modified"):
            # Process actual order modification in database
            await update_order_in_db_async(state["order_info"])
            return {
                "messages": [AIMessage(content="Order successfully modified! Is there anything else you need help with?")]
            }
        return {
            "messages": state["messages"]
        }
    
    # Add Nodes - wrap synchronous operations in asyncio.to_thread if needed
    builder.add_node("initialize", initialize_state)
    builder.add_node("modifier", order_modifier)
    builder.add_node("tools", tool_handler)
    
    # Add Edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "modifier")
    
    async def route_next(state: OrderState):
        if state.get("modified"):
            return "tools"
        if state.get("confirmation_pending"):
            return "modifier"
        return END
    
    builder.add_conditional_edges(
        "modifier",
        route_next,
        {
            "tools": "tools",
            "modifier": "modifier",
            END: END
        }
    )
    
    builder.add_edge("tools", END)
    
    # Use MemorySaver with async support
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
```

## Consumer
```python
import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .graph_system import create_order_support_graph, OrderState

class OrderSupportConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope['user']
        if not self.user.is_authenticated:
            await self.close()
            return
            
        self.conversation_id = self.scope['url_route']['kwargs'].get('conversation_id')
        # Create graph asynchronously
        self.graph = await create_order_support_graph()
        self.config = {
            "configurable": {
                "thread_id": self.conversation_id,
                "user_id": str(self.user.id)
            }
        }
        
        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome to Order Support, {self.user}!"
        }))

    async def receive(self, text_data=None):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'intent':
                await self.handle_intent(data)
            elif message_type == 'message':
                await self.handle_message(data)
            elif message_type == 'confirmation':
                await self.handle_confirmation(data)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def handle_intent(self, data):
        intent = data.get('intent')
        order_id = data.get('order_id')
        
        initial_state = {
            "messages": [HumanMessage(content=f"I need help with {intent} for order #{order_id}")],
            "order_id": order_id,
            "intent": intent,
            "conversation_id": self.conversation_id
        }
        
        try:
            async for event in self.graph.astream(initial_state, self.config):
                await self.process_graph_event(event)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing intent: {str(e)}"
            }))

    async def handle_message(self, data):
        message = data.get('message')
        
        try:
            async for event in self.graph.astream(
                {"messages": [HumanMessage(content=message)]},
                self.config
            ):
                await self.process_graph_event(event)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing message: {str(e)}"
            }))

    async def handle_confirmation(self, data):
        approved = data.get('approved')
        message = "confirm" if approved else "cancel"
        
        try:
            async for event in self.graph.astream(
                {"messages": [HumanMessage(content=message)]},
                self.config
            ):
                await self.process_graph_event(event)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing confirmation: {str(e)}"
            }))

    async def process_graph_event(self, event):
        if "messages" not in event:
            return
            
        message = event["messages"][-1]
        
        try:
            if isinstance(message, AIMessage):
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    await self.send(text_data=json.dumps({
                        'type': 'confirmation_required',
                        'message': message.content,
                        'tool_calls': message.tool_calls
                    }))
                else:
                    await self.send(text_data=json.dumps({
                        'type': 'assistant_response',
                        'message': message.content
                    }))
            
            elif isinstance(message, ToolMessage):
                await self.send(text_data=json.dumps({
                    'type': 'tool_response',
                    'message': message.content
                }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing event: {str(e)}"
            }))

```