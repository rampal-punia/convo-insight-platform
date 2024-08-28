from .models import UserMessage, Sentiment


def calculate_accuracy(ai_message):
    # This is a placeholder implementation. In a real-world scenario,
    # you might compare the AI's response to a gold standard or use user feedback.
    # For now, we'll use a random score as an example.
    import random
    return random.uniform(0.7, 1.0)


def calculate_relevance(ai_message):
    # In a real implementation, you might use NLP techniques to measure
    # the semantic similarity between the user's question and the AI's response.
    # For this example, we'll use a simple keyword matching approach.
    user_message = ai_message.in_reply_to
    if user_message and isinstance(user_message, UserMessage):
        common_words = set(user_message.content.lower().split()) & set(
            ai_message.content.lower().split())
        return len(common_words) / len(set(user_message.content.lower().split()))
    return 0.5  # default score if there's no user message to compare


def calculate_satisfaction(ai_message):
    # In a real-world scenario, this might be based on explicit user feedback.
    # For this example, we'll use the sentiment of the user's next message as a proxy for satisfaction.
    next_user_message = UserMessage.objects.filter(
        conversation=ai_message.conversation,
        created__gt=ai_message.created
    ).first()

    if next_user_message:
        sentiment = Sentiment.objects.filter(message=next_user_message).first()
        if sentiment:
            # Convert sentiment score to a satisfaction score (assuming sentiment ranges from -1 to 1)
            # This will give a score between 0 and 1
            return (sentiment.score + 1) / 2

    return 0.5  # default neutral score
