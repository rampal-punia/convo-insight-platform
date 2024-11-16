import json
from typing import Dict
from decimal import Decimal

import traceback
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from django.conf import settings
from django.db import transaction
from django.forms.models import model_to_dict
from .models import Order, OrderConversationLink, OrderItem
from convochat.models import Conversation, Message, UserText, AIText
from .utils import tool_manager as tm

import logging
logger = logging.getLogger('orders')  # Get the orders logger


class OrderSupportConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        logger.info(
            f"WebSocket connection attempt from user {self.scope['user']}")

        self.user = self.scope['user']
        self.message_count = 0
        self.summary_threshold = 5

        if not self.user.is_authenticated:
            logger.warning(f"Unauthenticated connection attempt")
            await self.close()
            return

        self.conversation_id = self.scope['url_route']['kwargs'].get(
            'conversation_id')
        await self.accept()

        logger.info(
            f"WebSocket connected for conversation {self.conversation_id}")

        await self.send(text_data=json.dumps({
            'type': 'welcome',
            'message': f"Welcome, {self.user}! You are now connected to Order Support."
        }))

        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
            request_timeout=30,
        )

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            logger.info(f"Received message type: {message_type}")
            logger.debug(f"Full message data: {data}")

            # Increment message count
            self.message_count += 1

            if message_type == 'intent':
                await self.handle_intent(data)
            elif message_type == 'message':
                await self.handle_message(data)
            elif message_type == 'confirmation':
                await self.handle_confirmation(data)
            else:
                logger.warning(
                    f"Unknown message type received: {message_type}")
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Unknown message type'
                }))

            # Check if we should summarize
            if self.message_count >= self.summary_threshold:
                await self.trigger_summarization()
                self.message_count = 0

        except Exception as e:
            logger.error(f"Error in receive: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing request: {str(e)}"
            }))

    async def handle_intent(self, data):
        """Handle intent selection"""
        try:
            intent = data.get('intent')
            order_id = data.get('order_id')
            conversation_id = data.get('uuid')

            logger.info(f"Processing intent: {intent} for order: {order_id}")

            # Get conversation and order details
            conversation, order = await self.get_or_create_conversation(conversation_id, order_id)
            # Store the intent
            conversation.current_intent = intent
            await database_sync_to_async(conversation.save)()
            order_details = await self.get_order_details(order_id)
            conversation_messages = await self.get_conversation_messages(conversation.id)

            # Initialize tools and graph
            tools = tm.create_order_tools(order_id)
            self.graph = self.create_intent_graph(
                intent, order_details, tools, conversation.id)

            # Create initial message
            user_message = await self.save_message(
                conversation=conversation,
                content_type='TE',
                is_from_user=True
            )
            await self.save_usertext(user_message, f"Request for {intent}")

            # Initialize state
            initial_state = {
                "messages": conversation_messages + [
                    HumanMessage(
                        content=f"I need help with {intent} for order #{order_id}.")
                ],
                "intent": intent,
                "order_info": order_details,
                "conversation_id": str(conversation.id)
            }

            logger.debug(f"Initial state: {initial_state}")

            # Process through graph
            async for event in self.graph.astream(initial_state):
                logger.debug(f"Graph event received: {event}")

                # Handle nested event structure
                if isinstance(event, dict):
                    messages = None
                    if "messages" in event:
                        messages = event["messages"]
                    elif "agent" in event and "messages" in event["agent"]:
                        messages = event["agent"]["messages"]

                    if not messages:
                        logger.warning(
                            f"Could not find messages in event: {event}")
                        continue

                    message = messages[-1]
                    logger.info(f"Processing message: {message}")

                    if isinstance(message, AIMessage):
                        content = message.content
                        logger.info(f"AI Message content: {content}")

                        # Save the AI response
                        ai_message = await self.save_message(
                            conversation=conversation,
                            content_type='TE',
                            is_from_user=False,
                            in_reply_to=user_message
                        )
                        await self.save_aitext(ai_message, content)

                        # Check for tool calls
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            await self.send(text_data=json.dumps({
                                'type': 'confirmation_required',
                                'action': content,
                                'tool_calls': [
                                    {"name": tc["name"], "args": tc["args"]}
                                    for tc in message.tool_calls
                                ]
                            }))
                        else:
                            # Send regular response
                            await self.send(text_data=json.dumps({
                                'type': 'agent_response',
                                'message': content
                            }))

                    elif isinstance(message, ToolMessage):
                        logger.info(f"Tool Message received: {message}")
                        # Send tool message to client if needed
                        await self.send(text_data=json.dumps({
                            'type': 'tool_response',
                            'message': message.content
                        }))
                    else:
                        logger.warning(
                            f"Unknown message type: {type(message)}")

        except Exception as e:
            logger.error(f"Error in handle_intent: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing intent: {str(e)}"
            }))

    async def trigger_summarization(self):
        '''Trigger conversation summarization'''
        try:
            # Create tools with current conversation context
            tools = await self.create_conversation_tools(self.conversation_id)

            # Find the summarization tool
            summarize_tool = next(
                tool for tool in tools
                if tool.name == "summarize_conversation"
            )

            # Execute summarization
            summary_result = await summarize_tool(
                self.conversation_id,
                max_length=150  # Adjust as needed
            )

            logger.info(f"Conversation summarized: {summary_result}")

            # Optionally notify client about summarization
            await self.send(text_data=json.dumps({
                'type': 'summary_updated',
                'summary': summary_result
            }))

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            logger.error(traceback.format_exc())

     # Add a method to manually trigger summarization

    async def handle_manual_summarization(self, data):
        """Handle manual summarization request"""
        try:
            max_length = data.get('max_length', 150)
            await self.trigger_summarization()

        except Exception as e:
            logger.error(f"Error handling manual summarization: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error generating summary: {str(e)}"
            }))

    async def handle_message(self, data):
        """Handle ongoing conversation"""
        try:
            conversation = await database_sync_to_async(Conversation.objects.get)(id=self.conversation_id)
            conversation_messages = await self.get_conversation_messages(conversation.id)
            user_input = data.get('message')
            order_id = data.get('order_id')

            # Get order details for context
            order_details = await self.get_order_details(order_id)

            logger.info(
                f"Processing message for order {order_id}: {user_input}")

            # Save user message
            user_message = await self.save_message(
                conversation=conversation,
                content_type='TE',
                is_from_user=True
            )
            await self.save_usertext(user_message, user_input)

            # Continue conversation with full context
            current_state = {
                "messages": conversation_messages + [HumanMessage(content=user_input)],
                "order_info": order_details,  # Add order info to state
                "conversation_id": str(conversation.id),
                # If you have intent stored, include it
                "intent": "modify_order"  # You might want to store this in the conversation model
            }

            logger.debug(f"Current state for processing: {current_state}")

            async for event in self.graph.astream(current_state):
                if event and ("messages" in event or "agent" in event):
                    # Handle nested event structure
                    messages = None
                    if "messages" in event:
                        messages = event["messages"]
                    elif "agent" in event and "messages" in event["agent"]:
                        messages = event["agent"]["messages"]

                    if not messages:
                        logger.warning(f"No messages found in event: {event}")
                        continue

                    message = messages[-1]
                    logger.info(f"Processing message: {message}")

                    # Save the AI response with proper tool calls handling
                    ai_message = await self.save_message(
                        conversation=conversation,
                        content_type='TE',
                        is_from_user=False,
                        in_reply_to=user_message
                    )

                    # Extract tool calls if they exist
                    tool_calls = getattr(message, 'tool_calls', [])
                    if tool_calls:
                        await self.save_aitext(ai_message, message.content, tool_calls)
                        await self.send(text_data=json.dumps({
                            'type': 'confirmation_required',
                            'action': message.content,
                            'tool_calls': [
                                {"name": tc["name"], "args": tc["args"]}
                                for tc in tool_calls
                            ]
                        }))
                    else:
                        await self.save_aitext(ai_message, message.content)
                        await self.send(text_data=json.dumps({
                            'type': 'agent_response',
                            'message': message.content
                        }))

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing message: {str(e)}"
            }))

    async def handle_confirmation(self, data):
        """Handle tool confirmation responses"""
        try:
            approved = data.get('approved')
            tool_calls = data.get('tool_calls', [])
            reason = data.get('reason', '')
            conversation_id = data.get('uuid')

            logger.info(
                f"Processing confirmation: approved={approved}, tool_calls={tool_calls}")

            # Get conversation and related data
            conversation = await database_sync_to_async(Conversation.objects.get)(id=conversation_id)
            conversation_messages = await self.get_conversation_messages(conversation.id)

            # Get the associated order from the tool calls
            order_id = None
            if tool_calls and len(tool_calls) > 0 and 'args' in tool_calls[0]:
                order_id = tool_calls[0]['args'].get('order_id')

            if not order_id:
                # Try to get order_id from the conversation link
                order_link = await database_sync_to_async(lambda: conversation.order_links.first())()
                if order_link:
                    order_id = order_link.order.id

            if not order_id:
                raise ValueError(
                    "Could not determine order ID for confirmation")

            # Get order details
            order_details = await self.get_order_details(order_id)

            current_state = {
                "messages": conversation_messages,
                "order_info": order_details,
                "conversation_id": str(conversation.id),
                "intent": conversation.current_intent
            }

            if approved:
                logger.info("User approved action, proceeding with tools")
                # Execute the approved action
                result = await self.execute_tool_action(tool_calls[0], order_details)

                # Update the conversation with the result
                ai_message = await self.save_message(
                    conversation=conversation,
                    content_type='TE',
                    is_from_user=False
                )
                await self.save_aitext(
                    ai_message,
                    f"Action completed successfully: {result}",
                    tool_calls=tool_calls
                )

                # Send completion notification
                await self.send(text_data=json.dumps({
                    'type': 'operation_complete',
                    'message': result,
                    'update_elements': [
                        {
                            'selector': '.card[data-order-id="{}"]'.format(order_id),
                            'html': await self.render_order_card(order_details)
                        }
                    ],
                    'completion_message': 'Order successfully updated!'
                }, default=self.decimal_default))

            else:
                logger.info(f"User declined action. Reason: {reason}")
                decline_message = ToolMessage(
                    content=f"Action denied by user. Reason: {reason}",
                    tool_call_id=tool_calls[0].get(
                        'id', 'unknown') if tool_calls else 'unknown'
                )
                current_state["messages"].append(decline_message)

                # Save the decline message
                ai_message = await self.save_message(
                    conversation=conversation,
                    content_type='TE',
                    is_from_user=False
                )
                await self.save_aitext(ai_message, decline_message.content)

                # Send decline notification
                await self.send(text_data=json.dumps({
                    'type': 'agent_response',
                    'message': f"Operation cancelled: {reason}"
                }, default=self.decimal_default))

        except Exception as e:
            logger.error(f"Error in handle_confirmation: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error processing confirmation: {str(e)}"
            }, default=self.decimal_default))

    async def create_conversation_tools(self, conversation_id: str):
        '''Create all conversation-related tools including summarization '''

        @tool
        async def summarize_conversation(conversation_id: str, max_length: int = 150) -> str:
            """Generate or update a summary of the current conversation."""
            try:
                # Get conversation and messages
                conversation = await database_sync_to_async(Conversation.objects.get)(id=conversation_id)
                messages = await database_sync_to_async(list)(
                    Conversation.messages.order_by('created').select_related(
                        'user_text', 'ai_text'
                    )
                )

                # Format conversation for summarization
                formatted_convo = "\n".join([
                    f"{'User' if msg.is_from_user else 'Assistant'}: "
                    f"{msg.user_text.content if msg.is_from_user else msg.ai_text.content}"
                    for msg in messages
                ])

                # Create summarization prompt
                current_summary = conversation.summary or ''
                if current_summary:
                    prompt = f"""Previous summary: {current_summary}

                            New conversation content:
                            {formatted_convo}
                            Please provide an updated summary incorporating the new content. Focus on key points, decisions made, and any important order details discussed. Keep the summary concise (maximum {max_length} words)."""
                else:
                    prompt = f"""Please summarize the following conversation, focusing on key points, decisions made, and any important order details discussed. Keep the summary concise (maximum {max_length} words):

                                {formatted_convo}"""

                # Generate summary using your LLM
                summary_message = await self.llm.ainvoke([
                    SystemMessage(
                        content="You are a helpful assistant that creates clear, concise summaries of customer service conversations."),
                    HumanMessage(content=prompt)
                ])

                # Update conversation summary
                @database_sync_to_async
                def update_summary():
                    with transaction.atomic():
                        conversation.summary = summary_message.content
                        conversation.save()

                await update_summary()

                return f"Summary updated: {summary_message.content}"

            except Conversation.DoesNotExist:
                return f"Conversation {conversation_id} not found."
            except Exception as e:
                logger.error(f"Error summarizing conversation: {str(e)}")
                return f"Error generating summary: {str(e)}"

    def create_intent_graph(self, intent: str, order_details: dict, tools: list, conversation_id: str) -> StateGraph:
        """Create a specialized graph based on the selected intent"""
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

        # Create prompt for selected intent
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get(intent, prompts["modify_order"])),
        ])

        # Create graph
        builder = StateGraph(tm.OrderState)

        # Bind tools to llm with explicit instructions
        llm_with_tools = self.llm.bind_tools(tools)

        # Create the chain by combining prompt and LLM
        chain = prompt | llm_with_tools

        async def agent(state: tm.OrderState) -> Dict:
            try:
                logger.debug(f"Agent processing state: {state}")

                # Check if we have a tool message (confirmation response)
                last_message = state["messages"][-1] if state["messages"] else None
                if isinstance(last_message, ToolMessage):
                    logger.info("Processing tool response")
                    # Format tool response for the LLM
                    tool_response = f"Previous action result: {last_message.content}"
                    state["messages"].append(
                        HumanMessage(content=tool_response))

                # Get conversation history
                messages = state["messages"]
                history = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    # Exclude the last message as it's the current input
                    for msg in messages[:-1]
                ])

                # Get latest input
                latest_message = messages[-1]
                user_input = latest_message.content if hasattr(
                    latest_message, "content") else str(latest_message)

                # Use the chain with the full context
                response = await chain.ainvoke({
                    "order_info": state["order_info"],
                    "conversation_history": history,
                    "user_input": user_input,
                })

                logger.info(f"LLM Response: {response}")

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

        # Add nodes and edges
        builder.add_node("agent", agent)
        builder.add_node('tools', ToolNode(tools))

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            'agent',
            tools_condition,
        )
        builder.add_edge('tools', 'agent')

        return builder.compile(interrupt_before=["tools"])

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
    def get_conversation_messages(self, conversation_id: str) -> list[dict]:
        """Fetch all messages for a conversation from the database"""
        conversation = Conversation.objects.get(id=conversation_id)
        messages = []
        for msg in conversation.messages.all().order_by('created'):
            if msg.is_from_user:
                messages.append(HumanMessage(content=msg.user_text.content))
            else:
                # For AI messages, only include tool_calls if they exist
                kwargs = {'content': msg.ai_text.content}
                if hasattr(msg.ai_text, 'tool_calls') and msg.ai_text.tool_calls:
                    kwargs['tool_calls'] = msg.ai_text.tool_calls
                messages.append(AIMessage(**kwargs))
        return messages

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
    def save_aitext(self, message, input_data, tool_calls=None):
        """Save AI text content with optional tool calls"""
        return AIText.objects.create(
            message=message,
            content=input_data,
            tool_calls=tool_calls if tool_calls else []
        )

    async def get_or_create_conversation(self, conversation_id, order_id):
        """Create or get existing conversation and link it to the order"""
        conversation, created = await database_sync_to_async(Conversation.objects.update_or_create)(
            id=conversation_id,
            defaults={
                'user': self.user,
                'status': 'AC'
            }
        )

        order = await database_sync_to_async(Order.objects.get)(id=order_id)
        await database_sync_to_async(OrderConversationLink.objects.get_or_create)(
            order=order,
            conversation=conversation
        )

        if created:
            await database_sync_to_async(
                Conversation.objects.filter(user=self.user, status='AC').exclude(
                    id=conversation_id).update
            )(status='EN')

        return conversation, order

    @database_sync_to_async
    def get_order_details(self, order_id, for_update=False):
        """Fetch order details with optional locking"""
        queryset = Order.objects.select_related('user')
        if for_update:
            queryset = queryset.select_for_update()

        order = queryset.get(id=order_id)
        order_dict = model_to_dict(order)
        items = order.items.all()
        item_dicts = [model_to_dict(item) for item in items]
        order_dict['items'] = item_dicts
        order_dict['status_description'] = order.get_status_display()
        order_dict['all_statuses'] = dict(Order.Status.choices)

        return json.loads(
            json.dumps(order_dict, default=self.decimal_default)
        )

    def decimal_default(self, obj):
        """Handle Decimal serialization"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def execute_tool_action(self, tool_call, order_details):
        """Execute the approved tool action"""
        tool_name = tool_call.get('name')
        tool_args = tool_call["args"]

        print("*"*40)
        print("Tool call is: ", tool_call)
        print("*"*40)

        if tool_name == 'modify_order_quantity':
            result = await self.update_order(
                order_id=order_details['id'],
                update_data={
                    'action': 'modify_quantity',
                    'item_id': tool_args.get('product_id'),
                    'new_quantity': tool_args.get('new_quantity')
                },
                order_details=order_details
            )
            return result

        elif tool_name == 'cancel_order_item':
            result = await self.update_order(
                order_id=order_details['id'],
                update_data={
                    'action': 'cancel_item',
                    'item_id': tool_args.get('item_id')
                },
                order_details=order_details
            )
            return result

        elif tool_name == 'cancel_order':
            result = await self.update_order(
                order_id=order_details['id'],
                update_data={
                    'action': 'cancel_order'
                },
                order_details=order_details
            )
            return result

        raise ValueError(f"Unknown tool: {tool_name}")

    @database_sync_to_async
    def update_order(self, order_id, update_data, order_details):
        try:
            with transaction.atomic():
                # Lock the order and related records
                order = Order.objects.select_for_update().get(id=order_id)

                # Validate user permissions
                if order.user != self.user:
                    raise PermissionError(
                        "Not authorized to modify this order")

                action = update_data.get('action')

                if action == 'modify_quantity':
                    # Get and lock the specific order item
                    order_item = OrderItem.objects.select_for_update().get(
                        product__id=update_data['item_id'],
                        order=order
                    )

                    # Validate stock availability
                    if update_data['new_quantity'] > order_item.product.stock:
                        raise ValueError(
                            f"Insufficient stock. Only {order_item.product.stock} available.")

                    # Store old quantity for price adjustment
                    old_quantity = order_item.quantity

                    # Update quantity and price
                    order_item.quantity = update_data['new_quantity']
                    order_item.price = order_item.product.price * \
                        update_data['new_quantity']
                    order_item.save()

                    # Update order total
                    order.total_amount = sum(
                        item.price for item in order.items.all()
                    )
                    order.save()

                    return {
                        'status': 'success',
                        'message': f"Quantity updated from {old_quantity} to {update_data['new_quantity']}",
                        'new_total': order.total_amount
                    }

                elif action == 'cancel_item':
                    # Get and lock the specific order item
                    order_item = OrderItem.objects.select_for_update().get(
                        product__id=update_data['item_id'],
                        order=order
                    )

                    # Store item details for confirmation message
                    item_name = order_item.product.name
                    item_quantity = order_item.quantity

                    # Delete the item
                    order_item.delete()

                    # Update order total
                    order.total_amount = sum(
                        item.price for item in order.items.all()
                    )

                    # If no items left, mark order as cancelled
                    if not order.items.exists():
                        order.status = Order.Status.CANCELLED

                    order.save()

                    return {
                        'status': 'success',
                        'message': f"Removed {item_quantity}x {item_name} from order",
                        'new_total': order.total_amount
                    }

                elif action == 'cancel_order':
                    # Can only cancel if order is in certain states
                    if order.status not in [Order.Status.PENDING, Order.Status.PROCESSING]:
                        raise ValueError(
                            "Order cannot be cancelled in its current state")

                    order.status = Order.Status.CANCELLED
                    order.save()

                    return {
                        'status': 'success',
                        'message': f"Order #{order.id} has been cancelled",
                        'new_status': order.get_status_display()
                    }

                else:
                    raise ValueError(f"Unknown action: {action}")

        except (Order.DoesNotExist, OrderItem.DoesNotExist):
            raise ValueError("Order or item not found")
        except Exception as e:
            logger.error(f"Error updating order: {str(e)}")
            raise

    @database_sync_to_async
    def render_order_card(self, order_details):
        """
        Render updated order card HTML for e-commerce order
        Includes order details, items, status, and total
        """

        items_html = ""
        for item in order_details['items']:
            items_html += f"""
                <div class="order-item mb-2">
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Product:</strong> {str(item['product'])}
                        </div>
                        <div class="col-md-2">
                            <strong>Qty:</strong> {str(item['quantity'])}
                        </div>
                        <div class="col-md-4">
                            <strong>Price:</strong> ${str(item['price'])}
                        </div>
                    </div>
                </div>
            """

        status_class = {
            'PE': 'text-warning',  # Pending
            'PR': 'text-info',     # Processing
            'SH': 'text-primary',  # Shipped
            'DE': 'text-success',  # Delivered
            'CA': 'text-danger',   # Cancelled
            'RT': 'text-secondary'  # Returned
        }.get(order_details['status'], '')

        return f"""
        <div class="card mb-3" data-order-id="{str(order_details['id'])}">
            <div class="card-header">
                <div class="row">
                    <div class="col-md-2">
                        <span class="text-body-secondary">ORDER ID</span><br>
                        Order #{str(order_details['id'])}
                    </div>
                    <div class="col-md-2">
                        <span class="text-body-secondary">ORDER PLACED</span><br>
                        {order_details['status']}
                    </div>
                    <div class="col-md-2">
                        <span class="text-body-secondary">TOTAL</span><br>
                        ${str(order_details['total_amount'])}
                    </div>
                    <div class="col-md-4">
                        <span class="text-body-secondary">STATUS</span><br>
                        <span class="{status_class}">{order_details['status_description']}</span>
                    </div>
                    <div class="col-md-2">
                        <span class="text-body-secondary">SHIP TO</span><br>
                        {order_details['user']}
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="order-items">
                    {items_html}
                </div>
            </div>
        </div>
        """
