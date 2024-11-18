# prompt_manager.py

from langchain_core.prompts import ChatPromptTemplate
from typing import Dict


class PromptManager:
    _prompts: Dict[str, ChatPromptTemplate] = {}

    @classmethod
    def initialize(cls):
        """Initialize all prompts"""
        cls._prompts = {
            'modify_order': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for order modifications. 
                When handling order changes:
                1. If a user requests a quantity change, confirm the specific details (product and new quantity)
                2. Use the modify_order_quantity tool with the following parameters:
                   - order_id: from order details
                   - customer_id: from user info
                   - product_id: from the item to modify
                   - new_quantity: the requested quantity
                3. Always confirm before executing the change
                4. Maintain conversation context
                
                Current context:
                Order details: {order_info}
                Previous conversation: {conversation_history}
                """),
                ("human", "{user_input}"),
            ]),
            'cancel_order': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for order cancellations.
                Current order details: {order_info}
                Previous conversation: {conversation_history}
                
                Follow these steps:
                1. Verify order eligibility for cancellation
                2. Confirm cancellation with user
                3. Use cancel_order tool with reason if confirmed
                """),
                ("human", "{user_input}"),
            ]),
            'order_status': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for order status inquiries.
                Current order details: {order_info}
                Previous conversation: {conversation_history}
                
                Provide clear status updates and use tracking tools when needed.
                """),
                ("human", "{user_input}"),
            ]),
            'delivery_issue': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for delivery issues.
                Current order details: {order_info}
                Previous conversation: {conversation_history}
                
                Handle delivery concerns professionally and use appropriate tools to resolve issues.
                """),
                ("human", "{user_input}"),
            ])
        }

    @classmethod
    def get_prompt(cls, intent: str) -> ChatPromptTemplate:
        """Get prompt template for given intent"""
        if not cls._prompts:
            cls.initialize()

        if intent not in cls._prompts:
            raise ValueError(f"No prompt template found for intent: {intent}")

        return cls._prompts[intent]
