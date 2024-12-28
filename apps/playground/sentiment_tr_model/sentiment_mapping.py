class EcommerceSentimentMapper:
    def __init__(self):
        # Define thresholds for confidence levels
        self.high_confidence = 0.75
        self.medium_confidence = 0.45
        self.low_confidence = 0.25

        # Define mappings for different confidence levels
        self.sentiment_mappings = {
            'joy': {
                'high': ['Delight', 'Very Satisfied'],
                'medium': ['Satisfied', 'Content'],
                'low': ['Slightly Positive', 'Neutral-Positive']
            },
            'love': {
                'high': ['Very Grateful', 'Highly Appreciative'],
                'medium': ['Grateful', 'Appreciative'],
                'low': ['Somewhat Pleased', 'Neutral-Positive']
            },
            'surprise': {
                'high': ['Amazed', 'Impressed'],
                'medium': ['Pleasantly Surprised', 'Intrigued'],
                'low': ['Neutral', 'Uncertain']
            },
            'sadness': {
                'high': ['Very Disappointed', 'Deeply Dissatisfied'],
                'medium': ['Disappointed', 'Dissatisfied'],
                'low': ['Slightly Disappointed', 'Neutral-Negative']
            },
            'anger': {
                'high': ['Very Frustrated', 'Outraged'],
                'medium': ['Frustrated', 'Annoyed'],
                'low': ['Slightly Frustrated', 'Displeased']
            },
            'fear': {
                'high': ['Urgent Concern', 'Serious Worry'],
                'medium': ['Concerned', 'Worried'],
                'low': ['Slight Concern', 'Uncertain']
            }
        }

        # Context-specific mappings for e-commerce situations
        self.context_mappings = {
            'shipping': {
                'joy': 'Delivery Satisfaction',
                'sadness': 'Delivery Disappointment',
                'anger': 'Shipping Frustration',
                'fear': 'Delivery Concern'
            },
            'product': {
                'joy': 'Product Satisfaction',
                'sadness': 'Product Disappointment',
                'anger': 'Product Complaint',
                'fear': 'Product Concern'
            },
            'service': {
                'joy': 'Service Satisfaction',
                'sadness': 'Service Disappointment',
                'anger': 'Service Complaint',
                'fear': 'Service Concern'
            },
            'payment': {
                'joy': 'Payment Success',
                'sadness': 'Payment Issue',
                'anger': 'Payment Frustration',
                'fear': 'Payment Concern'
            }
        }

    def get_confidence_level(self, score):
        """Determine confidence level based on score"""
        if score >= self.high_confidence:
            return 'high'
        elif score >= self.medium_confidence:
            return 'medium'
        elif score >= self.low_confidence:
            return 'low'
        return 'neutral'

    def map_sentiment(self, base_sentiment, confidence_score, context=None):
        """
        Map the base sentiment to e-commerce specific sentiment

        Args:
            base_sentiment (str): Original sentiment (joy, love, etc.)
            confidence_score (float): Confidence score from model
            context (str, optional): Specific context (shipping, product, etc.)

        Returns:
            dict: Mapped sentiment information
        """
        confidence_level = self.get_confidence_level(confidence_score)

        # Return neutral if confidence is too low
        if confidence_level == 'neutral':
            return {
                'sentiment': 'Neutral',
                'confidence_level': confidence_level,
                'score': confidence_score
            }

        # Get base sentiment mapping
        mapped_sentiments = self.sentiment_mappings.get(
            base_sentiment, {}).get(confidence_level, ['Neutral'])
        primary_sentiment = mapped_sentiments[0]

        # Apply context-specific mapping if available
        if context and context in self.context_mappings and base_sentiment in self.context_mappings[context]:
            primary_sentiment = self.context_mappings[context][base_sentiment]

        return {
            'sentiment': primary_sentiment,
            'alternative_sentiment': mapped_sentiments[1] if len(mapped_sentiments) > 1 else primary_sentiment,
            'confidence_level': confidence_level,
            'score': confidence_score,
            'base_sentiment': base_sentiment
        }

    def analyze_sentiment(self, text, base_sentiment, confidence_score):
        """
        Analyze text to determine context and map sentiment accordingly

        Args:
            text (str): Customer text to analyze
            base_sentiment (str): Original sentiment
            confidence_score (float): Confidence score

        Returns:
            dict: Analyzed sentiment with context
        """
        # Simple context detection based on keywords
        context = None
        text_lower = text.lower()

        if any(word in text_lower for word in ['delivery', 'shipping', 'shipment', 'arrive']):
            context = 'shipping'
        elif any(word in text_lower for word in ['product', 'item', 'order', 'purchase']):
            context = 'product'
        elif any(word in text_lower for word in ['service', 'support', 'help', 'assist']):
            context = 'service'
        elif any(word in text_lower for word in ['payment', 'refund', 'charge', 'price']):
            context = 'payment'

        # Get mapped sentiment with context
        sentiment_info = self.map_sentiment(
            base_sentiment, confidence_score, context)
        sentiment_info['context'] = context

        return sentiment_info


# Example usage:
if __name__ == "__main__":
    mapper = EcommerceSentimentMapper()

    # Example 1: High confidence joy in shipping context
    result1 = mapper.analyze_sentiment(
        "My package arrived earlier than expected!",
        "joy",
        0.85
    )

    # Example 2: Medium confidence anger in product context
    result2 = mapper.analyze_sentiment(
        "This product is not what I expected",
        "anger",
        0.60
    )

    # Example 3: Low confidence fear in payment context
    result3 = mapper.analyze_sentiment(
        "I'm not sure if my payment went through",
        "fear",
        0.30
    )

    print("Shipping Joy:", result1)
    print("Product Anger:", result2)
    print("Payment Fear:", result3)
