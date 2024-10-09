# from langchain_anthropic import ChatAnthropic
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode, tools_condition

# # Import your existing tools and add new ones for order-specific operations
# from existing_tools import fetch_user_flight_information, search_flights, lookup_policy
# from .models import Order, OrderItem


# def get_order_details(order_id):
#     order = Order.objects.get(id=order_id)
#     return {
#         "id": order.id,
#         "status": order.get_status_display(),
#         "total_amount": str(order.total_amount),
#         "items": [
#             {
#                 "product": item.product.name,
#                 "quantity": item.quantity,
#                 "price": str(item.price)
#             } for item in order.items.all()
#         ]
#     }


# def update_order_status(order_id, new_status):
#     order = Order.objects.get(id=order_id)
#     order.status = new_status
#     order.save()
#     return f"Order #{order_id} status updated to {order.get_status_display()}"


# def order_support_agent(order):
#     llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

#     tools = [
#         fetch_user_flight_information,
#         search_flights,
#         lookup_policy,
#         get_order_details,
#         update_order_status
#     ]

#     prompt = f"""You are a customer support AI for an e-commerce platform.
#     You're currently assisting with Order #{order.id}.
#     Use the provided tools to fetch order details and assist the customer.
#     Only update the order status if explicitly requested by the customer."""

#     runnable = prompt | llm.bind_tools(tools)

#     class State(TypedDict):
#         messages: Annotated[list[AnyMessage], add_messages]

#     workflow = StateGraph(State)

#     workflow.add_node("agent", runnable)
#     workflow.add_node("tools", ToolNode(tools))

#     workflow.set_entry_point("agent")
#     workflow.add_conditional_edges(
#         "agent",
#         tools_condition,
#         {
#             True: "tools",
#             False: END,
#         },
#     )
#     workflow.add_edge('tools', 'agent')

#     app = workflow.compile()

#     return app

# # Usage


# def run_agent(order, user_message):
#     agent = order_support_agent(order)
#     result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
#     return result['messages'][-1].content
