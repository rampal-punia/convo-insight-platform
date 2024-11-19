from langchain_core.prompts import ChatPromptTemplate
from typing import Dict


class PromptManager:
    _prompts: Dict[str, ChatPromptTemplate] = {}

    @classmethod
    def initialize(cls):
        """Initialize all prompts"""
        cls._prompts = {
            'track_order': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant specializing in order tracking and shipping updates.

                Order Details:
                {order_info}
                
                Tracking Information:
                - Current Status: {tracking_status}
                - Shipping Method: {shipping_method}
                - Estimated Delivery: {estimated_delivery}
                
                Previous conversation:
                {conversation_history}
                
                Follow these guidelines:
                1. First, confirm the order status and provide basic tracking information
                2. Use appropriate tracking tools based on the customer's specific query:
                - Use track_order for general tracking overview
                - Use get_tracking_details for detailed tracking history
                - Use get_shipment_location for current location
                - Use get_delivery_estimate for delivery timeline
                3. When discussing delivery dates:
                - Be specific about estimated delivery windows
                - Explain any potential delays
                - Provide confidence levels in estimates
                4. For tracking history:
                - Present updates chronologically
                - Highlight important status changes
                - Explain any unusual events
                5. Always maintain a helpful and informative tone"""),
                ("human", "{user_input}"),
            ]),

            'shipping_status': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for shipping and delivery updates.
            
                Order Details:
                {order_info}
                
                Current Tracking Status:
                - Status: {tracking_status}
                - Shipping Method: {shipping_method}
                - Estimated Delivery: {estimated_delivery}
                
                Previous conversation:
                {conversation_history}

                Guidelines for shipping status inquiries:
                1. First check current shipping status using get_tracking_details
                2. Explain shipping method and expected timeline
                3. Detail any transit updates or location changes
                4. Address any delays or exceptions
                5. Provide relevant carrier contact information"""),
                ("human", "{user_input}")
            ]),

            'delivery_estimate': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant providing delivery estimates and timelines.
            
                Order Details:
                {order_info}
                
                Current Delivery Information:
                - Status: {tracking_status}
                - Shipping Method: {shipping_method}
                - Current Estimated Delivery: {estimated_delivery}
                
                Previous conversation:
                {conversation_history}
                
                For delivery estimates:
                1. Use get_delivery_estimate tool for accurate timeline
                2. Consider shipping method and distance
                3. Account for any known delays
                4. Provide delivery window information
                5. Explain confidence level in estimate"""),
                ("human", "{user_input}"),
            ]),

            'modify_order_quantity': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for order modifications. 
                IMPORTANT: Before proceeding with any modification requests, first check the order status.
                Orders can ONLY be modified if they are in 'Pending' or 'Processing' status.
                Current order status is in order details.
                
                If the order status is not 'Pending' or 'Processing':
                1. Politely explain that the order cannot be modified due to its current status
                2. Provide information about what actions are possible in the current status
                3. Do not attempt to use any modification tools
                
                If the order status is 'Pending' or 'Processing', then:
                1. Confirm the specific details (product and new quantity)
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
                1. Verify order eligibility for cancellation. 
                2. Order status must be in ['PE', 'PR'] for cancellation
                3. Confirm cancellation with user
                4. Use cancel_order tool with reason if confirmed
                """),
                ("human", "{user_input}"),
            ]),
            'order_detail': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant for order details inquiries.
                
                Current order details: {order_info}
                Previous conversation: {conversation_history}
                
                Guidelines for order details:
                1. Provide clear order summary
                2. List individual items and quantities
                3. Show pricing information
                4. Include current order status
                5. Share shipping information if available
                
                When tracking is mentioned:
                1. Use track_order tool for status updates
                2. Provide delivery estimates when available
                3. Share carrier information
                4. Explain any status changes"""),
                ("human", "{user_input}"),
            ]),
            'delivery_issue': ChatPromptTemplate.from_messages([
                ("system", """You are a customer support assistant handling delivery issues and concerns.

                Current order details: {order_info}
                Previous conversation: {conversation_history}

                Follow these steps for delivery issues:
                1. Check current delivery status using get_tracking_details
                2. Identify any delivery exceptions or delays
                3. Explain the situation clearly to the customer
                4. Provide next steps or resolution options
                5. Use get_delivery_estimate for updated timeline

                Key considerations:
                - Address delivery concerns promptly
                - Explain any delays or exceptions
                - Provide alternative delivery options
                - Share carrier contact information
                - Offer to track or reroute if possible"""),
                ("human", "{user_input}"),
            ]),
        }

    @classmethod
    def get_prompt(cls, intent: str) -> ChatPromptTemplate:
        """Get prompt template for given intent"""
        if not cls._prompts:
            cls.initialize()

        if intent not in cls._prompts:
            raise ValueError(f"No prompt template found for intent: {intent}")

        return cls._prompts[intent]

    @classmethod
    def get_tool_description(cls, tool_name: str) -> str:
        """Get standardized description for tracking tools"""
        descriptions = {
            'track_order': """Get comprehensive tracking information for an order.
            Provides status, location, and estimated delivery.""",

            'get_tracking_details': """Get detailed tracking history and current status.
            Includes all tracking events and status updates.""",

            'get_shipment_location': """Get current location of the shipment.
            Provides latest known location and status.""",

            'get_delivery_estimate': """Get estimated delivery date and time window.
            Includes confidence level in estimate."""
        }
        return descriptions.get(tool_name, "Tool description not available.")

    @classmethod
    def customize_prompt(cls, intent: str, **kwargs) -> ChatPromptTemplate:
        """Customize prompt with additional context"""
        base_prompt = cls.get_prompt(intent)

        # Add any custom modifications based on kwargs
        if kwargs.get('tracking_focus'):
            # Add tracking-specific instructions
            system_message = base_prompt.messages[0].content
            tracking_addition = """
            Additional Tracking Guidelines:
            - Provide detailed status updates
            - Include location information
            - Share delivery timeframes
            - Explain any delays
            """
            new_system_message = system_message + tracking_addition
            return ChatPromptTemplate.from_messages([
                ("system", new_system_message),
                ("human", "{user_input}")
            ])

        return base_prompt

    @classmethod
    def format_tracking_response(cls, tracking_info: Dict) -> str:
        """Format tracking information in a standardized way"""
        return f"""
        Shipping Status: {tracking_info['status']}
        Current Location: {tracking_info.get('current_location', 'Not available')}
        
        Tracking Number: {tracking_info.get('tracking_number', 'Not available')}
        Carrier: {tracking_info.get('carrier', 'Not available')}
        
        Estimated Delivery: {tracking_info.get('estimated_delivery', 'Not available')}
        {f"Delivery Window: {tracking_info['delivery_window']}" if 'delivery_window' in tracking_info else ''}
        
        Latest Update: {tracking_info.get('latest_update', {}).get('description', 'No updates available')}
        """
