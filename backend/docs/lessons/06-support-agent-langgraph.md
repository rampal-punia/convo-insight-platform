# Lesson 6: Support Agent — LangGraph & Tool-Using AI

> How the LangGraph StateGraph powers the order support agent with tool calling, intent routing, and multi-turn conversations.

---

## What You'll Learn

- LangGraph StateGraph — building state machines for AI agents
- Tool definitions and the tool-calling loop
- Intent routing (hybrid: rule-based + LLM-based)
- Context management across conversation turns
- Flow management (greeting, support, closing)
- Prompt engineering for consistent behavior

---

## 1. What is LangGraph?

LangGraph is a framework for building **stateful multi-step AI agents**. Unlike a simple chain (prompt → LLM → output), an agent can:

1. Decide which tool to use
2. Call the tool
3. See the result
4. Decide what to do next (call another tool, respond to user, etc.)

### StateGraph concept:

```
                    ┌──────────┐
                    │ __start__ │
                    └─────┬─────┘
                          │
                          ▼
                    ┌──────────┐
              ┌────▶│ call_model│◀───────┐
              │     └─────┬─────┘        │
              │           │              │
              │    Should use tool?      │
              │      YES ↓    NO → END   │
              │    ┌──────┐              │
              │    │ tools │──────────────┘
              │    └──────┘    (tool results fed back)
              │
              └─── (loop continues until no more tools needed)
```

---

## 2. The Support Agent Graph

### Implementation (`apps/orders/graph_builder.py`):

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict

# Define the state (data passed between nodes)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    order_info: dict          # Current order details
    intent: str               # Detected user intent
    context: str              # Conversation context

def build_support_graph():
    workflow = StateGraph(AgentState)

    # Add nodes (processing steps)
    workflow.add_node("call_model", call_model_node)
    workflow.add_node("tools", tools_node)

    # Set entry point
    workflow.set_entry_point("call_model")

    # Add edges (transitions between nodes)
    workflow.add_conditional_edges(
        "call_model",
        should_use_tool,           # Decision function
        {
            "tools": "tools",       # If tool needed → go to tools node
            END: END,               # If no tool needed → end
        }
    )
    workflow.add_edge("tools", "call_model")  # After tool → back to model

    # Compile with memory (saves conversation state between turns)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
```

### The call_model node:

```python
async def call_model_node(state: AgentState):
    """Call the LLM to decide what to do."""
    llm = get_llm_with_tools()  # LLM bound with available tools
    response = await llm.ainvoke(state['messages'])
    return {"messages": [response]}
```

### The should_use_tool decision:

```python
def should_use_tool(state: AgentState) -> str:
    """Check if the LLM wants to use a tool."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"    # LLM requested a tool call
    return END            # LLM gave a direct response
```

### The tools node:

```python
async def tools_node(state: AgentState):
    """Execute the requested tool calls."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for call in tool_calls:
        tool = tool_map[call['name']]
        result = await tool.ainvoke(call['args'])
        results.append(ToolMessage(content=str(result), tool_call_id=call['id']))
    return {"messages": results}
```

---

## 3. Tool Definitions

Tools are functions the AI can call to perform actions.

### Tool Manager (`apps/orders/tool_manager.py`):

```python
from langchain_core.tools import tool

@tool
def modify_order_quantity(order_id: int, item_id: int, new_quantity: int) -> str:
    """Change the quantity of an item in an order.

    Args:
        order_id: The ID of the order to modify
        item_id: The ID of the order item to change
        new_quantity: The new quantity (must be >= 1)

    Returns:
        Confirmation message or error description
    """
    try:
        order = Order.objects.get(id=order_id)
        item = order.items.get(id=item_id)
        item.quantity = new_quantity
        item.save()
        return f"Updated item {item_id} quantity to {new_quantity} in order {order_id}"
    except Order.DoesNotExist:
        return f"Order {order_id} not found"

@tool
def check_order_status(order_id: int) -> str:
    """Check the current status of an order."""
    order = Order.objects.get(id=order_id)
    return f"Order {order_id} status: {order.status}. Items: {order.items.count()}"

@tool
def cancel_order(order_id: int) -> str:
    """Cancel an order if it hasn't been shipped yet."""
    order = Order.objects.get(id=order_id)
    if order.status in ['pending', 'processing']:
        order.status = 'cancelled'
        order.save()
        return f"Order {order_id} has been cancelled"
    return f"Cannot cancel order in status: {order.status}"

@tool
def get_order_details(order_id: int) -> str:
    """Get full details of an order including items and prices."""
    order = Order.objects.get(id=order_id)
    items = [f"- {item.product.name} x{item.quantity} @ ${item.price}" for item in order.items.all()]
    return f"Order #{order_id}\nStatus: {order.status}\nTotal: ${order.total_amount}\nItems:\n" + "\n".join(items)
```

### How tool calling works:

```
1. User: "I want to change item 3 in order 42 to quantity 5"

2. LLM receives: messages + list of available tools
   LLM decides: "I should call modify_order_quantity"

3. LLM output: ToolCall(name="modify_order_quantity",
                         args={"order_id": 42, "item_id": 3, "new_quantity": 5})

4. Tools node: Executes the function → "Updated item 3 quantity to 5 in order 42"

5. Tool result fed back to LLM

6. LLM generates final response: "Done! I've updated the quantity of item 3 to 5 in your order #42."
```

---

## 4. Hybrid Intent Router

The support agent uses a **hybrid** approach to classify user intent — combining fast rule-based matching with LLM fallback.

### Intent Router (`apps/support_agent/intent_router.py`):

```python
class HybridIntentRouter:
    def __init__(self):
        self.rule_patterns = {
            'order_status': [r'\b(status|track|where.*order|shipment)\b'],
            'modify_order': [r'\b(change|modify|update|quantity)\b'],
            'cancel_order': [r'\b(cancel|stop|abort)\b'],
            'return':       [r'\b(return|refund|money back)\b'],
            'greeting':     [r'\b(hi|hello|hey|good morning)\b'],
        }

    async def classify(self, text: str) -> dict:
        # Step 1: Try fast regex matching first
        for intent, patterns in self.rule_patterns.items():
            if any(re.search(p, text.lower()) for p in patterns):
                return {'intent': intent, 'confidence': 1.0, 'method': 'rule'}

        # Step 2: Fall back to LLM classification
        llm_result = await self._llm_classify(text)
        return {'intent': llm_result, 'confidence': 0.8, 'method': 'llm'}
```

**Why hybrid?**
- Regex is instant and free — handles obvious cases ("cancel my order", "hello")
- LLM handles ambiguous cases ("I don't want this anymore" → cancel vs return)
- Saves cost and latency on the easy cases

---

## 5. Context Manager

Keeps track of conversation state across multiple turns.

### Context Manager (`apps/support_agent/context_manager.py`):

```python
class ContextManager:
    async def build_context(self, conversation_id: int, user_message: str):
        # Get conversation history
        messages = await self.get_history(conversation_id)

        # Extract order reference from history or current message
        order_id = await self.extract_order_reference(messages, user_message)

        # Build structured context for the LLM
        context = {
            'conversation_id': conversation_id,
            'order_id': order_id,
            'order_info': await self.get_order_info(order_id) if order_id else None,
            'history_summary': await self.summarize_history(messages),
            'current_intent': await self.router.classify(user_message),
        }
        return context
```

### What context includes:
- **Order info**: Current order details (items, status, total)
- **History summary**: Condensed version of previous turns
- **Intent**: What the user wants to do
- **Entities**: Extracted order IDs, item numbers, quantities

---

## 6. Prompt Manager

### Prompt templates (`apps/support_agent/prompt_manager.py`):

```python
SUPPORT_SYSTEM_PROMPT = """You are a customer support agent for an e-commerce platform.

Current context:
- Order: {order_info}
- Customer intent: {intent}
- Conversation history: {history}

Guidelines:
1. Always confirm destructive actions (cancel, modify) before executing
2. Use tools to look up real order data — never guess
3. If the user's request is unclear, ask for clarification
4. Be empathetic for complaints, professional for inquiries
5. Keep responses concise and action-oriented

Available actions: check_order_status, modify_order_quantity, cancel_order, get_order_details
"""
```

---

## 7. Flow Manager

Controls the conversation flow through different phases.

### Flow Manager (`apps/support_agent/flow_manager.py`):

```python
class FlowManager:
    PHASES = ['greeting', 'identification', 'understanding', 'action', 'confirmation', 'closing']

    async def determine_phase(self, context):
        """Figure out which phase of the conversation we're in."""
        if not context.get('order_id'):
            return 'identification'    # Need to find the order
        if context['intent'] == 'greeting':
            return 'greeting'
        if context['intent'] in ['cancel_order', 'modify_order']:
            if not context.get('confirmed'):
                return 'confirmation'  # Need user to confirm
            return 'action'            # Execute the action
        return 'understanding'
```

---

## 8. Complete Request-Response Cycle

### Scenario: User wants to change order quantity

```
1. Client:  WS message → {"type": "text_message", "content": "Change item 2 in order 15 to qty 3"}

2. Consumer: receive_json()
   → Save user message to database

3. ContextManager: build_context()
   → Regex matches "change" → intent = "modify_order"
   → Extract order_id = 15 from text
   → Fetch order details from database

4. FlowManager: determine_phase()
   → order_id exists, intent = modify_order → phase = "action"
   → But not confirmed yet → phase = "confirmation"

5. LangGraph: call_model node
   → LLM sees: system prompt + context + user message + available tools
   → LLM response: "I'll change item 2 to quantity 3 in order #15. Should I proceed?"
   → No tool calls → END

6. Client receives: "I'll change item 2 to quantity 3 in order #15. Should I proceed?"

7. Client: "yes, please"

8. ContextManager: intent still = "modify_order", now confirmed
9. FlowManager: phase = "action"

10. LangGraph: call_model node
    → LLM decides to call modify_order_quantity(order_id=15, item_id=2, new_quantity=3)
    → Tool executes → "Updated item 2 quantity to 3 in order 15"
    → Back to call_model → LLM: "Done! Item 2 in order #15 is now set to quantity 3."

11. Client receives: "Done! Item 2 in order #15 is now set to quantity 3."
```

---

## 9. LangGraph vs Simple LangChain Chains

| Aspect | LangChain Chain | LangGraph Agent |
|--------|----------------|-----------------|
| Flow | Linear (prompt → LLM → output) | Cyclic (can loop back) |
| Tool use | Manual | Automatic (LLM decides) |
| State | Single request | Persistent across turns |
| Decision making | Static | Dynamic (conditional edges) |
| Use case | Simple Q&A | Complex multi-step tasks |

---

## Exercises

1. **Add a new tool** — Create a `apply_discount` tool that applies a percentage discount to an order. Add it to the tool manager and test it.
2. **Add sentiment-based routing** — If the user seems frustrated, route to a more empathetic prompt variant.
3. **Add conversation memory** — Use LangGraph's `MemorySaver` to persist conversation state in the database (not just in memory).

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/orders/graph_builder.py` | LangGraph StateGraph definition (~421 lines) |
| `apps/orders/tool_manager.py` | 5+ tool definitions (~517 lines) |
| `apps/orders/db_utils.py` | Database helper functions (~743 lines) |
| `apps/support_agent/intent_router.py` | Hybrid intent classifier (~528 lines) |
| `apps/support_agent/context_manager.py` | Conversation context builder (~405 lines) |
| `apps/support_agent/flow_manager.py` | Conversation phase controller (~264 lines) |
| `apps/support_agent/prompt_manager.py` | Prompt templates |
| `apps/orders/views.py` | 5 CBVs for order management |
| `apps/orders/tasks.py` | Celery tasks (intent, sentiment) |
