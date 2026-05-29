# Intern Assignment 07: Wire the Report Issue Button — Delivery Issue Tools & Resolution Flow

**Track:** Backend — LangGraph Tools + Django Models + WebSocket Consumer
**Difficulty:** Intermediate-Advanced
**Estimated Effort:** 5–6 hours
**Prerequisites:** Understand the order detail page (read the template + consumer), comfortable with LangChain tools

---

## Problem Statement

The order detail page at `/orders/<id>/` has a **"Report Issue"** button that sends the `delivery_issue` intent over WebSocket. The prompt exists and the agent can read tracking information, but it **cannot actually resolve or report any problems**:

```python
# apps/orders/utils/tool_manager.py — ReturnRequest schema exists but is NEVER used
class ReturnRequest(BaseOrderSchema):
    """Schema for return requests"""
    items: List[dict] = Field(description="List of items to return")
    reason: str = Field(description="Reason for return")
    condition: str = Field(description="Condition of items being returned")
```

The schema is defined, imported nowhere, and never turned into a tool. When a user clicks "Report Issue" and describes a problem like:

> "My package arrived damaged" or "I received the wrong item" or "I want to return this"

The agent can only say "Let me check the tracking for you" — it has **no tools to**:
- Create a support ticket / issue record
- Request a return
- Request a refund
- Escalate to a human agent
- Log the issue in the database

---

## Root Cause Analysis

### What Exists

| Component | Status |
|-----------|--------|
| `delivery_issue` prompt in `PromptManager` | Exists — tells agent to check tracking |
| `ReturnRequest` schema in `tool_manager.py` | Exists — but never registered as a tool |
| Tracking tools (`get_tracking_details`, `get_delivery_estimate`) | Work — but only READ data |
| **Issue/Ticket model** | **Missing** — nowhere to store reported issues |
| **`report_issue` tool** | **Missing** — can't create issue records |
| **`request_return` tool** | **Missing** — ReturnRequest schema unused |
| **`request_refund` tool** | **Missing** — no refund capability |
| **Consumer issue handling** | **Missing** — `construct_tool_input` has no issue cases |

### The Flow Today (Broken)

```
User clicks "Report Issue"
  → WebSocket sends {type: 'intent', intent: 'delivery_issue', order_id: 12}
  → Consumer creates graph with intent='delivery_issue'
  → Graph loads the delivery_issue prompt
  → Agent can only call tracking tools (track_order, get_tracking_details)
  → Agent tells user "Your package is in transit" — USELESS
  → User says "I want a refund"
  → Agent has no tool to process this → stalls or hallucinates
```

### The Flow It Should Have

```
User clicks "Report Issue"
  → Agent checks tracking (existing tools)
  → Agent identifies the issue
  → Agent asks what resolution the user wants
  → User says "I want a return"
  → Agent calls request_return tool → creates issue record → shows confirmation
  → Consumer sends 'confirmation_required' → user confirms → return processed
```

---

## Assignment Tasks

### Task 1: Create the Issue Model (30 min)

**File:** `apps/orders/models.py`

Add an `OrderIssue` model to track reported issues:

```python
class OrderIssue(models.Model):
    """Track delivery issues, return requests, and refund requests."""

    class IssueType(models.TextChoices):
        DAMAGED = 'DM', 'Damaged'
        WRONG_ITEM = 'WI', 'Wrong Item'
        MISSING_ITEM = 'MI', 'Missing Item'
        LATE_DELIVERY = 'LD', 'Late Delivery'
        NOT_RECEIVED = 'NR', 'Not Received'
        OTHER = 'OT', 'Other'

    class ResolutionType(models.TextChoices):
        RETURN = 'RT', 'Return'
        REFUND = 'RF', 'Refund'
        REPLACEMENT = 'RP', 'Replacement'
        ESCALATE = 'ES', 'Escalate to Agent'

    class IssueStatus(models.TextChoices):
        REPORTED = 'RP', 'Reported'
        IN_REVIEW = 'IR', 'In Review'
        APPROVED = 'AP', 'Approved'
        RESOLVED = 'RS', 'Resolved'
        REJECTED = 'RJ', 'Rejected'

    order = models.ForeignKey(
        Order, on_delete=models.CASCADE,
        related_name='issues',
    )
    issue_type = models.CharField(
        max_length=2, choices=IssueType.choices, default=IssueType.OTHER,
    )
    description = models.TextField(blank=True)
    resolution_requested = models.CharField(
        max_length=2, choices=ResolutionType.choices, blank=True,
    )
    status = models.CharField(
        max_length=2, choices=IssueStatus.choices, default=IssueStatus.REPORTED,
    )
    resolution_notes = models.TextField(blank=True)
    items_json = models.JSONField(
        default=list, blank=True,
        help_text="List of affected items [{product_id, quantity, condition}]",
    )
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
```

After creating the model:
1. Create and run the migration: `python manage.py makemigrations orders && python manage.py migrate`
2. Register it in `apps/orders/admin.py`

---

### Task 2: Add DB Methods for Issues (30 min)

**File:** `apps/orders/utils/db_utils.py`

Add these async methods to `DatabaseOperations`:

```python
@database_sync_to_async
def create_issue(self, order_id: str, issue_type: str, description: str,
                 resolution_requested: str = '', items: list = None) -> dict:
    """Create a new issue report for an order."""
    try:
        order = Order.objects.get(id=order_id, user=self.user)
        issue = OrderIssue.objects.create(
            order=order,
            issue_type=issue_type,
            description=description,
            resolution_requested=resolution_requested,
            items_json=items or [],
        )
        return {
            'issue_id': issue.id,
            'issue_type': issue.get_issue_type_display(),
            'status': issue.get_status_display(),
            'resolution': issue.get_resolution_requested_display(),
            'created': issue.created.isoformat(),
        }
    except Order.DoesNotExist:
        return {"error": "Order not found"}
    except Exception as e:
        logger.error(f"Error creating issue: {e}")
        return {"error": str(e)}


@database_sync_to_async
def get_order_issues(self, order_id: str) -> list:
    """Get all issues for an order."""
    issues = OrderIssue.objects.filter(
        order_id=order_id, order__user=self.user,
    ).order_by('-created')
    return [
        {
            'issue_id': i.id,
            'type': i.get_issue_type_display(),
            'status': i.get_status_display(),
            'description': i.description,
            'resolution': i.get_resolution_requested_display(),
            'created': i.created.isoformat(),
        }
        for i in issues
    ]


@database_sync_to_async
def process_return(self, order_id: str, items: list, reason: str, condition: str) -> dict:
    """Process a return request for specific items."""
    try:
        order = Order.objects.get(id=order_id, user=self.user)
        if order.status not in [Order.Status.DELIVERED, Order.Status.IN_TRANSIT]:
            return {
                "error": f"Cannot return items for order in '{order.get_status_display()}' status. "
                         "Returns are only available for delivered or in-transit orders."
            }

        # Create issue record
        issue = OrderIssue.objects.create(
            order=order,
            issue_type=OrderIssue.IssueType.OTHER,
            description=f"Return request: {reason}",
            resolution_requested=OrderIssue.ResolutionType.RETURN,
            items_json=items,
        )

        return {
            'status': 'success',
            'issue_id': issue.id,
            'message': f"Return request created for {len(items)} item(s). "
                       f"Issue #{issue.id} is being reviewed.",
        }
    except Order.DoesNotExist:
        return {"error": "Order not found"}
    except Exception as e:
        logger.error(f"Error processing return: {e}")
        return {"error": str(e)}


@database_sync_to_async
def process_refund(self, order_id: str, reason: str, amount: str = None) -> dict:
    """Process a refund request."""
    try:
        order = Order.objects.get(id=order_id, user=self.user)
        refund_amount = float(amount) if amount else float(order.total_amount)

        issue = OrderIssue.objects.create(
            order=order,
            issue_type=OrderIssue.IssueType.OTHER,
            description=f"Refund request: {reason}",
            resolution_requested=OrderIssue.ResolutionType.REFUND,
        )

        return {
            'status': 'success',
            'issue_id': issue.id,
            'refund_amount': refund_amount,
            'message': f"Refund request for ${refund_amount:.2f} created. "
                       f"Issue #{issue.id} is being reviewed.",
        }
    except Order.DoesNotExist:
        return {"error": "Order not found"}
    except Exception as e:
        logger.error(f"Error processing refund: {e}")
        return {"error": str(e)}
```

**Important:** Add the import for `OrderIssue` at the top of the file:
```python
from ..models import (
    Order,
    OrderConversationLink,
    OrderItem,
    OrderTracking,
    OrderIssue,   # ← Add this
)
```

---

### Task 3: Create Issue Resolution Tools (45 min)

**File:** `apps/orders/utils/tool_manager.py`

**3a.** Create the tool schemas:

```python
class ReportIssueRequest(BaseOrderSchema):
    """Schema for reporting a delivery issue"""
    issue_type: str = Field(
        description="Type of issue: DM (Damaged), WI (Wrong Item), MI (Missing Item), "
                    "LD (Late Delivery), NR (Not Received), OT (Other)"
    )
    description: str = Field(description="Detailed description of the issue")
    resolution_requested: str = Field(
        default="",
        description="Requested resolution: RT (Return), RF (Refund), RP (Replacement), ES (Escalate)"
    )
```

**3b.** Add tool creation methods to `ToolManager`:

```python
def _create_report_issue_tool(self):
    """Create the report delivery issue tool"""

    @tool(args_schema=ReportIssueRequest)
    async def report_delivery_issue(
        order_id: str, customer_id: str,
        issue_type: str, description: str,
        resolution_requested: str = "",
    ) -> str:
        """Report a delivery issue for an order. Creates an issue record in the system."""
        if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
            return "Unable to process request due to too many nested tool calls"

        self.tool_call_depth += 1
        self.current_tool_chain.append("report_delivery_issue")

        try:
            result = await self.db_ops.create_issue(
                order_id, issue_type, description, resolution_requested,
            )
            if "error" in result:
                return f"Error: {result['error']}"

            return (
                f"Issue reported successfully!\n"
                f"Issue ID: #{result['issue_id']}\n"
                f"Type: {result['issue_type']}\n"
                f"Status: {result['status']}\n"
                f"Requested Resolution: {result['resolution']}\n"
                f"Created: {result['created']}"
            )
        finally:
            self.tool_call_depth -= 1
            self.current_tool_chain.pop()

    return report_delivery_issue


def _create_request_return_tool(self):
    """Create the request return tool"""

    @tool(args_schema=ReturnRequest)
    async def request_return(
        order_id: str, customer_id: str,
        items: list, reason: str, condition: str,
    ) -> str:
        """Request a return for items in an order. Only available for delivered or in-transit orders."""
        if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
            return "Unable to process request due to too many nested tool calls"

        self.tool_call_depth += 1
        self.current_tool_chain.append("request_return")

        try:
            result = await self.db_ops.process_return(
                order_id, items, reason, condition,
            )
            if "error" in result:
                return f"Error: {result['error']}"

            return result['message']
        finally:
            self.tool_call_depth -= 1
            self.current_tool_chain.pop()

    return request_return


def _create_request_refund_tool(self):
    """Create the request refund tool"""

    @tool(args_schema=BaseOrderSchema)
    async def request_refund(order_id: str, customer_id: str, reason: str = "") -> str:
        """Request a refund for an order. Creates a refund request for review."""
        if self.tool_call_depth >= self.MAX_TOOL_DEPTH:
            return "Unable to process request due to too many nested tool calls"

        self.tool_call_depth += 1
        self.current_tool_chain.append("request_refund")

        try:
            result = await self.db_ops.process_refund(
                order_id, reason,
            )
            if "error" in result:
                return f"Error: {result['error']}"

            return result['message']
        finally:
            self.tool_call_depth -= 1
            self.current_tool_chain.pop()

    return request_refund
```

**3c.** Register the new tools in `_register_tools`:

```python
def _register_tools(self):
    # ... existing tool registrations ...

    # Register issue resolution tools
    self.registry.register(
        self._create_report_issue_tool(),
        ToolMetadata(
            category=ToolCategory.SAFE,
            description="Report a delivery issue",
            requires_confirmation=False,
        ),
    )

    self.registry.register(
        self._create_request_return_tool(),
        ToolMetadata(
            category=ToolCategory.SENSITIVE,
            description="Request a return for order items",
            requires_confirmation=True,
        ),
    )

    self.registry.register(
        self._create_request_refund_tool(),
        ToolMetadata(
            category=ToolCategory.SENSITIVE,
            description="Request a refund for an order",
            requires_confirmation=True,
        ),
    )
```

**3d.** Update the `tool_categories` dict in `ToolRegistry.__init__`:

```python
self.tool_categories = {
    # ... existing categories ...
    "issue": ["report_delivery_issue"],
    "return": ["request_return"],
    "refund": ["request_refund"],
}
```

---

### Task 4: Update the Delivery Issue Prompt (20 min)

**File:** `apps/orders/utils/prompt_manager.py`

Replace the `delivery_issue` prompt with a comprehensive one that knows about the new tools:

```python
'delivery_issue': ChatPromptTemplate.from_messages([
    ("system", """You are a customer support assistant handling delivery issues and concerns.

    Current order details: {order_info}
    Previous conversation: {conversation_history}

    Follow these steps for delivery issues:

    Step 1 — IDENTIFY THE PROBLEM:
    - Ask the customer what went wrong if not clear
    - Check tracking using get_tracking_details if relevant
    - Identify the issue type: damaged, wrong item, missing item, late, not received

    Step 2 — SUGGEST RESOLUTION:
    Based on the issue, suggest one of:
    - For damaged/wrong/missing items → Use request_return (requires confirmation)
    - For unsatisfied customer wanting money back → Use request_refund (requires confirmation)
    - For tracking/delivery questions → Use get_delivery_estimate or get_shipment_location
    - For any issue → Use report_delivery_issue to log it in the system

    Step 3 — EXECUTE:
    - If the customer wants a return: collect items, reason, condition → call request_return
    - If the customer wants a refund: collect reason → call request_refund
    - For all issues: call report_delivery_issue to create a record
    - Always explain what will happen next (review timeline, email confirmation, etc.)

    Key rules:
    - NEVER promise instant refunds or returns — all go through review
    - ALWAYS create an issue record, even if just tracking a complaint
    - Be empathetic — delivery issues are frustrating for customers
    - If the issue is complex or the customer is very upset, suggest escalation
    """),
    ("human", "{user_input}"),
]),
```

---

### Task 5: Update Consumer to Handle Issue Tool Confirmations (30 min)

**File:** `apps/orders/consumers.py`

**5a.** Add `request_return` and `request_refund` to `construct_tool_input`:

```python
elif tool_name == 'request_return':
    return {
        'order_id': str(tool_args.get('order_id')),
        'customer_id': str(tool_args.get('customer_id')),
        'items': tool_args.get('items', []),
        'reason': tool_args.get('reason', ''),
        'condition': tool_args.get('condition', 'good'),
    }

elif tool_name == 'request_refund':
    return {
        'order_id': str(tool_args.get('order_id')),
        'customer_id': str(tool_args.get('customer_id')),
        'reason': tool_args.get('reason', 'Customer requested refund'),
    }

elif tool_name == 'report_delivery_issue':
    return {
        'order_id': str(tool_args.get('order_id')),
        'customer_id': str(tool_args.get('customer_id')),
        'issue_type': tool_args.get('issue_type', 'OT'),
        'description': tool_args.get('description', ''),
        'resolution_requested': tool_args.get('resolution_requested', ''),
    }
```

**5b.** The confirmation flow already handles sensitive tools via `interrupt_before=["sensitive_tools"]` in the graph builder. Since `request_return` and `request_refund` are registered as `SENSITIVE`, they will automatically trigger the confirmation dialog on the frontend. No additional changes needed for the confirmation mechanism.

---

### Task 6: Write Tests (45 min)

**File:** `apps/orders/tests/test_issue_tools.py` (create the directory and `__init__.py` if needed)

```python
import pytest
from django.test import override_settings
from django.contrib.auth import get_user_model

from orders.models import Order, OrderItem, OrderIssue, OrderTracking
from products.models import Product, Category

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="issueuser", password="testpass")


@pytest.fixture
def category(db):
    return Category.objects.create(name="Electronics", slug="electronics")


@pytest.fixture
def product(db, category):
    return Product.objects.create(
        name="Wireless Headphones",
        slug="wireless-headphones",
        price=99.99,
        stock=50,
        category=category,
    )


@pytest.fixture
def delivered_order(user, product):
    order = Order.objects.create(
        user=user,
        status=Order.Status.DELIVERED,
        total_amount=99.99,
    )
    OrderItem.objects.create(
        order=order, product=product, quantity=1, price=99.99,
    )
    OrderTracking.objects.create(
        order=order,
        status=OrderTracking.TrackingStatus.DELIVERED,
        location="Customer Address",
        description="Package delivered",
    )
    return order


@pytest.mark.django_db
class TestOrderIssueModel:
    def test_create_issue(self, delivered_order):
        issue = OrderIssue.objects.create(
            order=delivered_order,
            issue_type=OrderIssue.IssueType.DAMAGED,
            description="Product arrived with cracked case",
            resolution_requested=OrderIssue.ResolutionType.RETURN,
        )
        assert issue.id is not None
        assert issue.status == OrderIssue.IssueStatus.REPORTED
        assert issue.get_issue_type_display() == "Damaged"

    def test_issue_string_representation(self, delivered_order):
        issue = OrderIssue.objects.create(
            order=delivered_order,
            issue_type=OrderIssue.IssueType.WRONG_ITEM,
            description="Got blue instead of red",
        )
        assert "Wrong Item" in issue.get_issue_type_display()


@pytest.mark.django_db
class TestCreateIssueDBOperation:
    @pytest.mark.asyncio
    async def test_create_issue_success(self, user, delivered_order):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.create_issue(
            order_id=str(delivered_order.id),
            issue_type="DM",
            description="Product arrived damaged",
            resolution_requested="RT",
        )
        assert "issue_id" in result
        assert result["issue_type"] == "Damaged"
        assert result["status"] == "Reported"

    @pytest.mark.asyncio
    async def test_create_issue_invalid_order(self, user):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.create_issue(
            order_id="99999",
            issue_type="DM",
            description="Test",
        )
        assert "error" in result


@pytest.mark.django_db
class TestProcessReturn:
    @pytest.mark.asyncio
    async def test_return_delivered_order(self, user, delivered_order, product):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.process_return(
            order_id=str(delivered_order.id),
            items=[{"product_id": product.id, "quantity": 1, "condition": "damaged"}],
            reason="Product arrived damaged",
            condition="damaged",
        )
        assert result["status"] == "success"
        assert "Return request created" in result["message"]

    @pytest.mark.asyncio
    async def test_return_pending_order_rejected(self, user, product):
        from orders.utils.db_utils import DatabaseOperations

        order = Order.objects.create(
            user=user, status=Order.Status.PENDING, total_amount=99.99,
        )
        OrderItem.objects.create(order=order, product=product, quantity=1, price=99.99)

        db_ops = DatabaseOperations(user)
        result = await db_ops.process_return(
            order_id=str(order.id),
            items=[{"product_id": product.id, "quantity": 1}],
            reason="Changed mind",
            condition="good",
        )
        assert "error" in result


@pytest.mark.django_db
class TestProcessRefund:
    @pytest.mark.asyncio
    async def test_refund_delivered_order(self, user, delivered_order):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.process_refund(
            order_id=str(delivered_order.id),
            reason="Product not as described",
        )
        assert result["status"] == "success"
        assert "Refund request" in result["message"]

    @pytest.mark.asyncio
    async def test_refund_custom_amount(self, user, delivered_order):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.process_refund(
            order_id=str(delivered_order.id),
            reason="Partial refund for damaged packaging",
            amount="49.99",
        )
        assert result["status"] == "success"
        assert result["refund_amount"] == 49.99
```

Cover:
- `OrderIssue` model creation and status defaults
- `create_issue` DB method — success and error cases
- `process_return` — delivered order accepted, pending order rejected
- `process_refund` — full and partial refund amounts
- Tool invocation through `ToolManager` — returns properly formatted strings

---

## File Reference Map

```
backend/
├── apps/orders/
│   ├── models.py                        ← Task 1 (add OrderIssue model)
│   ├── admin.py                         ← Task 1 (register OrderIssue)
│   ├── utils/
│   │   ├── db_utils.py                  ← Task 2 (add issue DB methods)
│   │   ├── tool_manager.py              ← Task 3 (add 3 new tools)
│   │   └── prompt_manager.py            ← Task 4 (update delivery_issue prompt)
│   ├── consumers.py                     ← Task 5 (add tool input construction)
│   └── tests/
│       └── test_issue_tools.py          ← Task 6 (create)
└── docs/lessons/assignments/
    └── 07_report_delivery_issue_tools.md  ← this file
```

---

## Key Concepts to Learn

1. **LangChain tool schemas** — Pydantic `BaseModel` classes define tool parameters; `args_schema` links them to `@tool` functions
2. **Sensitive vs Safe tools** — Returns and refunds are `SENSITIVE` (require user confirmation); issue reporting is `SAFE`
3. **Django model enums** — `TextChoices` for type/status fields with display names
4. **`interrupt_before` in LangGraph** — The graph pauses before executing sensitive tools, letting the consumer ask for user confirmation
5. **Tool categorization pattern** — `ToolRegistry` + `ToolMetadata` for managing tool permissions

---

## Submission Checklist

- [ ] `OrderIssue` model created with `IssueType`, `ResolutionType`, `IssueStatus` enums
- [ ] Migration created and applied successfully
- [ ] `OrderIssue` registered in admin with list_display, list_filter
- [ ] `create_issue`, `get_order_issues`, `process_return`, `process_refund` DB methods work
- [ ] `report_delivery_issue` tool works and creates issue records
- [ ] `request_return` tool works (sensitive, requires confirmation)
- [ ] `request_refund` tool works (sensitive, requires confirmation)
- [ ] All 3 tools registered in `ToolRegistry` with correct categories
- [ ] `delivery_issue` prompt updated with tool documentation
- [ ] Consumer `construct_tool_input` handles all new tools
- [ ] All existing tests pass
- [ ] New tests in `test_issue_tools.py` pass

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
