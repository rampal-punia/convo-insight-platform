# Capstone Project: IISC CDS Course

## Domain: Customer Service and Conversational AI

- **Techniques**: Natural Language Processing (NLP), Large Language Models (LLMs), Sentiment Analysis, Intent Recognition, Topic Modeling
- **Title**: Customer Conversational Intelligence Platform Powered by an LLM Agent

## Overview and Problem Statement:

This project aims to develop a state-of-the-art Customer Conversational Intelligence Platform powered by a Large Language Model (LLM) agent. The LLM's advanced language understanding will drive the analysis of customer interactions across diverse channels (chatbots, call centers, email, social media). The platform will extract actionable insights from this data, enabling businesses to optimize customer service processes and significantly enhance the overall customer experience.

## Datasets:

### 1.

- **Name**: Relational Strategies in Customer Interactions (RSiCS)

- **Description**: This dataset contains a corpus for improving the quality and relational abilities of Intelligent Virtual Agents.

- **Link**: [Link to the dataset](https://www.kaggle.com/datasets/veeralakrishna/relational-strategies-in-customer-servicersics)

### 2.

- **Name**: 3K Conversations Dataset for ChatBot from Kaggle
- **Description**: The dataset includes various types of conversations such as casual or formal discussions, interviews, customer service interactions, and social media conversations.
- **Link**: [Link to the dataset](https://www.kaggle.com/datasets/kreeshrajani/3k-conversations-dataset-for-chatbot)

### 3.

- **Name**: Customer Support on Twitter Dataset from Kaggle
- **Description**: This is a large corpus of tweets and replies that can aid in natural language understanding and conversational models.
- **Link**: [Link to the dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

## Challenges:

- **Data Collection**: Gather customer conversations from diverse sources like voice calls, chat transcripts, emails, and social media interactions.

- **Use LLM-Agent for**:

    - a. **Sentiment Analysis** – accurate detection of customer emotions (positive, negative, neutral) and granular sentiment categories (frustration, satisfaction, inquiry, etc.) throughout conversations.

    - b. **Intent Recognition** - understanding the underlying purpose behind customers' queries, enabling tailored responses and resolutions.

    - c. **Topic Modeling** - discovering recurring themes and patterns within conversations, highlighting trending issues, feedback topics, and potential areas for improvement.

    - d. Agent Performance Evaluation - Analyzing agent interactions to provide constructive feedback, identifying training needs, and recognizing exceptional service.

    - e. LLM-Driven Real-time Recommendations - Empowering agents with suggestions for next-best actions or responses during active conversations, optimizing outcomes.

## Methodology:

Select GPT2/GPT3, fine-tune the LLM agent extensively on a large dataset of customer conversations annotated for sentiment, intent, topics, etc. Develop ML algorithms to support the LLM agent. The primary focus will be on the LLM's ability to perform sentiment analysis, intent recognition, topic modeling, and agent performance assessment. Utilize platforms like SageMaker or equivalent to automate the ML workflow.

## Example:

### Customer: 
Hello, I ordered a laptop from your website, and it's been a week, but I haven't received it yet. Can you help me track my order?

### Platform Analysis :

- **Categorization**: Inquiry about order tracking.
- **Sentiment Analysis**: Neutral sentiment.
- **Resolution Status**: Unresolved.
- **Support Agent**: Hi there! I apologize for the delay in your order. Could you please provide me with your order number? I'll check the status for you.

### Customer: 
My order number is 123456789.

### Platform Analysis:

- **Categorization**: Providing order information.
- **Sentiment Analysis**: Neutral sentiment.
- **Resolution Status**: In progress.
- **Support Agent**: Thank you for providing the order number. Let me check that for you.

*[Platform sends a real-time request to the order tracking system]*

### Platform Analysis:

- **Real-time Analysis**: The platform receives updated order tracking information. The laptop is currently in transit and is expected to arrive in two days.

#### Response Generation
- **Support Agent**: Good news! Your laptop is on its way and should be delivered within the next two days. Here's your tracking number: ABC123XYZ. You can use this number to monitor its progress.

### Customer:
Thank you for the information. I appreciate your help.

### Platform Analysis:
- **Sentiment Analysis**: Positive sentiment.
- **Resolution Status**: Resolved.
- **Support Agent**: You're welcome! If you have any more questions or need further assistance, feel free to ask. Have a great day!

## Significance:

As companies accumulate immense volumes of customer interaction data, the ability to unlock meaningful insights and streamline customer service processes becomes a competitive advantage. The envisioned platform, with its real-time analysis capabilities, has the potential to revolutionize customer service, ultimately translating into greater customer satisfaction, increased operational efficiency, and a strengthened market position for businesses.

## Reference:
1. Conversational Health Agents: A Personalized LLM-Powered Agent Framework, Mahyar Abbasian, Iman Azimi, Amir M. Rahmani, Ramesh Jain: https://arxiv.org/html/2310.02374v4

2. Building a Conversational AI Agent with Long-Term Memory Using LangChain and Milvus, Zilliz:
https://medium.com/@zilliz_learn/building-a-conversational-ai-agent-with-long-term-memory-using-langchain-and-milvus-0c4120ad7426



This is a captone project details for the one year long Advance certification course on computrational datascience. I want you to create a nice effective and cachy title for this project and a readme file. This readme file will explain all the steps and strategies taken to complete this project. I am planning use the following tools and libraries. the django, django-channels, celery, postgres, pgvectore, asgi, wsgi, langchain for this project.