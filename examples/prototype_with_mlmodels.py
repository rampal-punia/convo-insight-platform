import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
import spacy
import xgboost as xgb
from gensim import corpora
from gensim.models import LdaMulticore
import dill

# Initialize spaCy and NLTK
nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def analyze_sentiment(text):
    return random.uniform(-1, 1)


def detect_intent(text):
    intents = ['Question', 'Complaint', 'Request', 'Feedback']
    return random.choice(intents)


def extract_topics(text):
    topics = ['Product Features', 'Pricing',
              'Technical Issues', 'Customer Service']
    return random.sample(topics, k=random.randint(1, 3))


@st.cache_resource
def load_models():
    try:
        # Load sentiment analysis models
        sentiment_word2vec = joblib.load(
            'ml_models/sentiment_ml/word2vec_model_train.joblib')
        sentiment_model = joblib.load(
            'ml_models/sentiment_ml/xgboost_classifier_model_train.joblib')

        # Load intent recognition models
        intent_components = joblib.load(
            'ml_models/intent_ml/intent_recognition_components.joblib')

        # Load topic modeling components
        lda_model = joblib.load('ml_models/topic_ml/lda_model.joblib')
        dictionary = joblib.load('ml_models/topic_ml/dictionary.joblib')
        with open('ml_models/topic_ml/preprocess_text_function.dill', 'rb') as f:
            topic_preprocess = dill.load(f)

        return {
            'sentiment': {
                'word2vec': sentiment_word2vec,
                'model': sentiment_model
            },
            'intent': intent_components,
            'topic': {
                'lda_model': lda_model,
                'dictionary': dictionary,
                'preprocess': topic_preprocess
            }
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def get_custom_stopwords():
    nltk_stop_words = set(nltk.corpus.stopwords.words('english'))
    spacy_stop_words = nlp.Defaults.stop_words

    negation_words = {
        'no', 'not', 'nor', 'neither', 'never', 'none',
        "n't", 'cannot', "couldn't", "didn't", "doesn't",
        "hadn't", "hasn't", "haven't", "isn't", "mightn't",
        "mustn't", "needn't", "shan't", "shouldn't", "wasn't",
        "weren't", "won't", "wouldn't"
    }

    combined_stopwords = nltk_stop_words.union(spacy_stop_words)
    return combined_stopwords - negation_words


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    custom_stopwords = get_custom_stopwords()
    tokens = [token for token in tokens if token.isalpha()
              and token not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def analyze_text_with_models(text, models):
    # Sentiment Analysis
    def get_sentiment(text, models):
        processed_text = preprocess_text(text)
        embeddings = [models['sentiment']['word2vec'].wv[word]
                      for word in processed_text
                      if word in models['sentiment']['word2vec'].wv]
        if embeddings:
            doc_vector = np.mean(embeddings, axis=0)
            dpredict = xgb.DMatrix([doc_vector])
            pred = models['sentiment']['model']['classifier'].predict(dpredict)[
                0]
            sentiment = models['sentiment']['model']['label_encoder'].inverse_transform([
                                                                                        int(pred)])[0]
            return sentiment, pred
        return 0, 0

    # Intent Recognition
    def get_intent(text, models):
        processed_text = preprocess_text(text)
        embedding = np.mean([models['intent']['word2vec_model'].wv[word]
                             for word in processed_text
                             if word in models['intent']['word2vec_model'].wv], axis=0)
        tfidf_features = models['intent']['tfidf_vectorizer'].transform([
                                                                        text]).toarray()
        combined_features = np.hstack((embedding, tfidf_features[0]))
        dpredict = xgb.DMatrix([combined_features])
        pred = int(models['intent']['classifier'].predict(dpredict)[0])
        intent = models['intent']['label_encoder'].inverse_transform([pred])[0]
        return intent

    # Topic Analysis
    def get_topics(text, models):
        processed_text = models['topic']['preprocess'](text)
        bow = models['topic']['dictionary'].doc2bow(processed_text)
        topics = models['topic']['lda_model'].get_document_topics(bow)
        return sorted(topics, key=lambda x: x[1], reverse=True)

    sentiment, sentiment_score = get_sentiment(text, models)
    intent = get_intent(text, models)
    topics = get_topics(text, models)

    return {
        'sentiment': sentiment,
        'sentiment_score': sentiment_score,
        'intent': intent,
        'topics': topics
    }


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

# Main Streamlit App
st.set_page_config(page_title="ConvoInsight Prototype", layout="wide")
st.title("ConvoInsight: Customer Conversational Intelligence Platform")

# Load models once
models = load_models()

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["Dashboard", "Chat Interface", "Conversation Analysis", "ML Analysis Playground"]
)

if page == "ML Analysis Playground":
    st.header("ML Analysis Playground")

    text_input = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="Type or paste your text here for comprehensive analysis..."
    )

    if st.button("Analyze Text", type="primary"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
        else:
            try:
                with st.spinner("Analyzing text..."):
                    results = analyze_text_with_models(text_input, models)

                    # Display results in columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### Sentiment Analysis")
                        sentiment_emoji = "😊" if results['sentiment'] == 1 else "😐" if results['sentiment'] == 0 else "😞"
                        st.markdown(f"**Sentiment:** {sentiment_emoji}")
                        st.markdown(
                            f"**Score:** {results['sentiment_score']:.2f}")

                    with col2:
                        st.markdown("### Intent Recognition")
                        st.markdown(
                            f"**Detected Intent:** {results['intent']}")

                    with col3:
                        st.markdown("### Topic Analysis")
                        for topic_id, prob in results['topics'][:3]:
                            topic_terms = models['topic']['lda_model'].show_topic(
                                topic_id, 3)
                            terms = ", ".join(
                                [term for term, _ in topic_terms])
                            st.markdown(f"**Topic {topic_id}** ({prob:.2f})")
                            st.markdown(f"Keywords: {terms}")

                    # Visualizations
                    st.markdown("### Visualizations")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Topic distribution plot
                        topic_probs = [prob for _, prob in results['topics']]
                        topic_ids = [f"Topic {id}" for id,
                                     _ in results['topics']]
                        fig = px.bar(
                            x=topic_ids,
                            y=topic_probs,
                            title="Topic Distribution"
                        )
                        st.plotly_chart(fig)

                    with col2:
                        # Sentiment gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=results['sentiment_score'],
                            title={'text': "Sentiment Score"},
                            gauge={'axis': {'range': [-1, 1]},
                                   'bar': {'color': "darkblue"},
                                   'steps': [
                                {'range': [-1, -0.3], 'color': "red"},
                                {'range': [-0.3, 0.3], 'color': "gray"},
                                {'range': [0.3, 1], 'color': "green"}
                            ]}
                        ))
                        st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

elif page == "Dashboard":
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

            # Use ML models for analysis instead of random generation
            if models:
                analysis = analyze_text_with_models(user_input, models)

                ai_response = "Thank you for your message. Our team will get back to you shortly."
                st.text_area("AI:", ai_response, height=100)

                st.subheader("Message Analysis")
                st.write(
                    f"Sentiment: {analysis['sentiment']} ({analysis['sentiment_score']:.2f})")
                st.write(f"Detected Intent: {analysis['intent']}")
                st.write("Top Topics:")
                for topic_id, prob in analysis['topics'][:3]:
                    topic_terms = models['topic']['lda_model'].show_topic(
                        topic_id, 3)
                    terms = ", ".join([term for term, _ in topic_terms])
                    st.write(f"- Topic {topic_id} ({prob:.2f}): {terms}")

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
st.sidebar.write(
    "This is a prototype of the ConvoInsight platform with integrated ML capabilities for sentiment analysis, intent recognition, and topic modeling.")


# cd examples
# streamlit run prototype_with_mlmodels.py
