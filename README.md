# ConvoInsight: Customer Conversational Intelligence Platform

ConvoInsight is a state-of-the-art Customer Conversational Intelligence Platform powered by a Large Language Model (LLM) agent. This platform analyzes customer interactions across diverse channels to extract actionable insights, enabling businesses to optimize customer service processes and enhance the overall customer experience.

### Note:

*This project is in active development and is currently in its initial phase.*

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [API Documentation](#api-documentation)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

## Project Overview

ConvoInsight leverages advanced Natural Language Processing (NLP) techniques and Large Language Models to analyze customer conversations from various sources such as chatbots, call centers, emails, and social media. The platform provides real-time insights, including sentiment analysis, intent recognition, topic modeling, and agent performance evaluation.

### Key Objectives

- Develop a robust conversational intelligence platform using Django and Django Channels
- Implement advanced NLP techniques using LangChain and custom LLM models
- Provide real-time analysis and recommendations for customer service interactions
- Create a scalable and efficient system using Celery for background task processing
- Utilize PostgreSQL with pgvector for efficient storage and retrieval of conversation data and embeddings

## Features

- Multi-channel conversation analysis (chat, voice, email, social media)
- Real-time sentiment analysis
- Intent recognition
- Topic modeling and trend identification
- Agent performance evaluation
- LLM-driven real-time recommendations for customer service agents
- Interactive dashboards for insights visualization
- Scalable architecture for handling large volumes of conversations

## Technology Stack

- **Backend Framework**: Django
- **Asynchronous Support**: Django Channels
- **Database**: PostgreSQL with pgvector extension
- **Task Queue**: Celery with Redis as message broker
- **LLM Integration**: LangChain
- **Web Servers**: ASGI (Daphne) for WebSocket, WSGI (Gunicorn) for HTTP
- **Frontend**: Intial implementation (Djanog-templates with JS and Jquery) way forward with React.js
- **Containerization**: Docker (optional, for easy deployment)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rampal-punia/convo-insight-platform.git
   cd convo-insight-platform
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up PostgreSQL and create a database for the project.

5. Install the pgvector extension in your PostgreSQL database.

6. Set up environment variables (create a `.env` file in the project root):
   ```
   DEBUG=True
   SECRET_KEY=your_secret_key
   DATABASE_URL=postgresql://user:password@localhost/dbname
   REDIS_URL=redis://localhost:6379
   ```

7. Run migrations:
   ```
   python manage.py migrate
   ```

8. Start the development server:
   ```
   python manage.py runserver
   ```

## Usage

1. Access the admin interface at `http://localhost:8000/admin/` to manage data and configurations.

2. Use the API endpoints to interact with the platform programmatically. Refer to the [API Documentation](#api-documentation) section for details.

3. Access the frontend interface at `http://localhost:8000/` to view dashboards and insights.

## Project Structure

```bash
convo-insight-platform/
├── convoinsight/          # Main Django project directory
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── apps/
│   ├── analysis/          # App for conversation analysis
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── tasks.py       # Celery tasks for real-time analysis
│   │   └── utils/         # Utility functions for analysis
│   ├── dashboard/         # App for user interface and visualizations
│   │   ├── models.py
│   │   └── views.py
│   └── api/               # App for API endpoints
│       ├── models.py
│       └── views.py
├── ml_models/             # Directory for ML model development
│   ├── notebooks/         # Jupyter notebooks for model development
│   │   ├── sentiment_analysis_1.ipynb
│   │   ├── sentiment_analysis_2.ipynb
│   │   ├── intent_recognition.ipynb
│   │   ├── topic_modeling.ipynb
│   │   └── agent_performance.ipynb
│   ├── training_scripts/  # Scripts for model training
│   │   ├── train_sentiment_model.py
│   │   ├── train_intent_model.py
│   │   ├── train_topic_model.py
│   │   └── train_performance_model.py
│   └── saved_models/      # Directory to store trained models
├── data_processing/       # Scripts for data ingestion and preprocessing
│   ├── ingest_chat_data.py
│   ├── ingest_sentiments.py
│   ├── ingest_voice_data.py
│   ├── ingest_email_data.py
│   └── ingest_social_media_data.py
├── static/                # Static files (CSS, JS, images)
├── templates/             # HTML templates
├── tests/                 # Test cases
├── manage.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Documentation

Detailed API documentation can be found at `/api/docs/` when running the server. This includes all endpoints, request/response formats, and authentication requirements.

Key endpoints include:
- `/api/conversations/`: Upload and retrieve conversation data
- `/api/analysis/sentiment/`: Get sentiment analysis for conversations
- `/api/analysis/intent/`: Get intent recognition results
- `/api/analysis/topics/`: Get topic modeling results
- `/api/recommendations/`: Get real-time recommendations for agents

## Contributing

We welcome contributions to ConvoInsight! Please follow these steps to contribute [more details](https://github.com/rampal-punia/convo-insight-platform/blob/master/CONTRIBUTING.md):

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure your code adheres to our coding standards and include tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project is part of the "Advanced Certification Course on Computational Data Science at [IISC](https://iisc.ac.in/)" with [TalentSpirit](https://talentsprint.com/course/computational-data-science-iisc-bangalore). 

- We utilize the following datasets:
  - Relational Strategies in Customer Interactions (RSiCS)
  - 3K Conversations Dataset for ChatBot from Kaggle
  - Customer Support on Twitter Dataset from Kaggle
