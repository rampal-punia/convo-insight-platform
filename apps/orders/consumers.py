import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from decimal import Decimal
from typing import Annotated, TypedDict, Literal

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from apps.convochat.utils import configure_llm
from convochat.models import Conversation, Message, UserText, AIText
from django.forms.models import model_to_dict
from .models import Order, OrderConversationLink


class OrderState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    order_info: dict


class OrderSupportConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope['user']
        if not self.user.is_authenticated:
            await self.close()
            return

        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')
        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! You are now connected to Order Support."
        }))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None):
        """Handle incoming WebSocket messages"""
        data = json.loads(text_data)
        intent = data.get('intent')
        order_id = data.get('order_id')
        conversation_id = data.get('uuid')

        # Get conversation and order details
        conversation, order = await self.get_or_create_conversation(conversation_id, order_id)
        order_details = await self.get_order_details(order_id)

        # Get conversation history
        history = await self.get_conversation_history(conversation.id)

        # Create and save initial message
        user_message = await self.save_message(
            conversation=conversation,
            content_type='TE',
            # content=f"Intent: {intent}",
            is_from_user=True
        )
        await self.save_usertext(user_message, f"Request for {intent}")

        # Create and process the intent-specific graph
        graph = self.create_intent_graph(intent, order_details)
        config = {"configurable": {"thread_id": str(conversation.id)}}

        # Initialize state with intent and order information
        initial_state = {
            "messages": history + [("system", f"Handle {intent} request for order {order_id}")],
            "intent": intent,
            "order_info": order_details
        }

        # Process through graph and send response
        # try:
        result = graph.invoke(initial_state, config)
        if result and "messages" in result:
            ai_message = await self.save_message(
                conversation=conversation,
                content_type='TE',
                # content=result["messages"][-1].content,
                is_from_user=False,
                in_reply_to=user_message
            )
            await self.save_aitext(ai_message, result["messages"][-1].content)

            await self.send(text_data=json.dumps({
                'type': 'agent_response',
                'message': result["messages"][-1].content
            }))
        # except Exception as e:
        #     await self.send(text_data=json.dumps({
        #         'type': 'error',
        #         'message': f"Error processing request: {str(e)}"
        #     }))

    def create_intent_graph(self, intent: str, order_details: dict) -> StateGraph:
        """Create a specialized graph based on the selected intent"""
        prompts = {
            "order_status": """You are an order status assistant. Use the order details to provide accurate 
                             status updates, estimated delivery times, and tracking information.
                             Current order details: {order_details}
                             
                             Conversation history:
                             {history}
                             
                             Handle the following request: {messages[-1].content}""",

            "return_request": """You are a returns specialist. Review the order details and eligibility for returns.
                               Provide clear instructions for the return process, including timeframes and conditions.
                               Current order details: {order_details}
                               
                               Conversation history:
                               {history}
                               
                               Handle the following request: {messages[-1].content}""",

            "modify_order": """You are an order modification specialist. Check if the order can still be modified
                             and explain what changes are possible at this stage.
                             Current order details: {order_details}
                             
                             Conversation history:
                             {history}
                             
                             Handle the following request: {messages[-1].content}""",

            "delivery_issue": """You are a delivery support specialist. Address delivery concerns and issues,
                               providing solutions and next steps based on the order status.
                               Current order details: {order_details}
                               
                               Conversation history:
                               {history}
                               
                               Handle the following request: {messages[-1].content}"""
        }

        # Create the prompt for the selected intent
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get(intent, prompts["order_status"]))
        ])

        # Create the graph
        builder = StateGraph(OrderState)

        # Define the agent node
        def agent(state: OrderState):
            llm = configure_llm.LLMConfig.get_llm(
                model_name='gpt-4o-mini',
                model_provider='openai'
            )
            response = llm.invoke(state["messages"])
            print("response from model is: ", response)
            print("state['messages'] is: ", state["messages"])
            return {"messages": [response]}

        # Add nodes and edges
        builder.add_node("agent", agent)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)

        # Compile with memory
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    @database_sync_to_async
    def get_conversation_history(self, conversation_id, limit=8):
        """Fetch conversation history"""
        conversation = Conversation.objects.get(id=conversation_id)
        messages = conversation.messages.order_by('-created')[:limit]
        return [
            HumanMessage(content=msg.user_text.content) if msg.is_from_user else AIMessage(
                content=msg.ai_text.content)
            for msg in reversed(messages)
        ]

    @database_sync_to_async
    def save_message(self, conversation, content_type, is_from_user=True, in_reply_to=None):
        """Save a message to the database"""
        return Message.objects.create(
            conversation=conversation,
            content_type=content_type,
            is_from_user=is_from_user,
            in_reply_to=in_reply_to
        )

    @database_sync_to_async
    def save_usertext(self, message, input_data):
        """Save user text content"""
        return UserText.objects.create(
            message=message,
            content=input_data,
        )

    @database_sync_to_async
    def save_aitext(self, message, input_data):
        """Save AI text content"""
        return AIText.objects.create(
            message=message,
            content=input_data,
        )

    @database_sync_to_async
    def get_or_create_conversation(self, conversation_id, order_id):
        """Create or get existing conversation and link it to the order"""
        conversation, created = Conversation.objects.update_or_create(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )
        order = Order.objects.get(id=order_id)
        OrderConversationLink.objects.get_or_create(
            order=order,
            conversation=conversation
        )
        if created:
            Conversation.objects.filter(user=self.user, status='AC').exclude(
                id=conversation_id
            ).update(status='EN')
        return conversation, order

    @database_sync_to_async
    def get_order_details(self, order_id):
        """Fetch and format order details"""
        order = Order.objects.get(id=order_id)
        order_dict = model_to_dict(order)
        items = order.items.all()
        item_dicts = [model_to_dict(item) for item in items]
        order_dict['items'] = item_dicts
        order_dict['status_description'] = order.get_status_display()
        order_dict['all_statuses'] = dict(Order.Status.choices)

        # Convert to JSON-compatible format
        json_compatible_dict = json.loads(
            json.dumps(order_dict, default=self.decimal_default)
        )
        return json_compatible_dict

    def decimal_default(self, obj):
        """Handle Decimal serialization"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
