import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random


def generate_conversation():
    return {
        'id': random.randint(1000, 9999),
        'title': f"Conversation {random.randint(1, 100)}",
        'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
        'sentiment': random.uniform(-1, 1),
        'topic': random.choice(['Product Inquiry', 'Technical Support', 'Billing Issue', 'Feedback']),
        'status': random.choice(['Active', 'Ended'])
    }


conversations = [generate_conversation() for _ in range(50)]


def analyze_sentiment(text):
    return random.uniform(-1, 1)


def detect_intent(text):
    intents = ['Question', 'Complaint', 'Request', 'Feedback']
    return random.choice(intents)


def extract_topics(text):
    topics = ['Product Features', 'Pricing',
              'Technical Issues', 'Customer Service']
    return random.sample(topics, k=random.randint(1, 3))


# Streamlit app
st.set_page_config(page_title="ConvoInsight Prototype", layout="wide")

st.title("ConvoInsight: Customer Conversational Intelligence Platform")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Navigate", ["Dashboard", "Chat Interface", "Conversation Analysis"])

if page == "Dashboard":
    st.header("Dashboard")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Conversations", len(conversations))
    col2.metric("Active Conversations", len(
        [c for c in conversations if c['status'] == 'Active']))
    col3.metric("Avg. Sentiment",
                f"{sum(c['sentiment'] for c in conversations) / len(conversations):.2f}")
    col4.metric("Most Common Topic", max(set(c['topic'] for c in conversations), key=lambda x: sum(
        1 for c in conversations if c['topic'] == x)))

    # Conversations over time
    df = pd.DataFrame(conversations)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    fig = px.line(daily_counts, x='date', y='count',
                  title='Conversations Over Time')
    st.plotly_chart(fig)

    # Sentiment distribution
    fig = px.histogram(df, x='sentiment', title='Sentiment Distribution')
    st.plotly_chart(fig)

    # Topic distribution
    topic_counts = df['topic'].value_counts()
    fig = px.pie(values=topic_counts.values,
                 names=topic_counts.index, title='Topic Distribution')
    st.plotly_chart(fig)

elif page == "Chat Interface":
    st.header("Chat Interface")

    user_input = st.text_input("Type your message here:")
    if st.button("Send"):
        if user_input:
            st.text_area("User:", user_input, height=100)

            # Simulated analysis
            sentiment = analyze_sentiment(user_input)
            intent = detect_intent(user_input)
            topics = extract_topics(user_input)

            ai_response = "Thank you for your message. Our team will get back to you shortly."
            st.text_area("AI:", ai_response, height=100)

            st.subheader("Message Analysis")
            st.write(f"Sentiment: {sentiment:.2f}")
            st.write(f"Detected Intent: {intent}")
            st.write(f"Extracted Topics: {', '.join(topics)}")

elif page == "Conversation Analysis":
    st.header("Conversation Analysis")

    # Select a conversation
    selected_conv = st.selectbox("Select a conversation", [
                                 f"Conversation {c['id']}" for c in conversations])
    conv = next(
        c for c in conversations if f"Conversation {c['id']}" == selected_conv)

    st.subheader(f"Analysis for {selected_conv}")
    st.write(f"Date: {conv['timestamp']}")
    st.write(f"Status: {conv['status']}")
    st.write(f"Topic: {conv['topic']}")
    st.write(f"Sentiment: {conv['sentiment']:.2f}")

    # Simulated conversation content
    messages = [
        {"role": "user", "content": "Hi, I'm having issues with my account."},
        {"role": "assistant", "content": "I'm sorry to hear that. Can you please provide more details about the issue you're experiencing?"},
        {"role": "user", "content": "I can't log in. It says my password is incorrect, but I'm sure it's right."},
        {"role": "assistant", "content": "I understand. Let's try to resolve this. First, can you confirm that you're using the correct email address for your account?"},
    ]

    for msg in messages:
        st.text_area(msg['role'].capitalize(), msg['content'], height=100)
        if msg['role'] == 'user':
            st.write(f"Sentiment: {analyze_sentiment(msg['content']):.2f}")
            st.write(f"Intent: {detect_intent(msg['content'])}")
            st.write(f"Topics: {', '.join(extract_topics(msg['content']))}")
        st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.write("This is a prototype of the ConvoInsight platform. It demonstrates key features such as conversation analysis, sentiment tracking, and intent recognition.")
