# ConvoInsight: Customer Conversational Intelligence Platform

ConvoInsight is a state-of-the-art Customer Conversational Intelligence Platform powered by a Large Language Model (LLM) agent. This platform analyzes customer interactions across diverse channels to extract actionable insights, enabling businesses to optimize customer service processes and enhance the overall customer experience.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API Documentation](#api-documentation)
8. [Contributing](#contributing)
9. [Datasets](#datasets)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

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
- Real-time sentiment analysis with granular emotion detection
- Intent recognition for tailored responses
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
- **Frontend**: Initial implementation (Django templates with JS and jQuery), future plans for React.js
- **Containerization**: Docker (for deployment)
- **Machine Learning**: Custom fine-tuned LLM models, SageMaker for ML workflow automation

## Project Structure

```
convo-insight-platform/
├── config/                # Main Django project directory
├── apps/                  # Django apps (accounts, analysis, convochat, dashboard, api)
├── ml_models/             # ML model development and training scripts
├── data_processing/       # Scripts for data ingestion and preprocessing
├── utils/                 # Shared utility functions
├── static/                # Static files (CSS, JS, images)
├── templates/             # HTML templates
├── tests/                 # Test cases
├── manage.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

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

6. Set up environment variables (create a `.env` file in the project root).

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

2. Use the API endpoints to interact with the platform programmatically.

3. Access the frontend interface at `http://localhost:8000/` to view dashboards and insights.

## API Documentation

Detailed API documentation can be found at `/api/docs/` when running the server. Key endpoints include:
- `/api/conversations/`: Upload and retrieve conversation data
- `/api/analysis/sentiment/`: Get sentiment analysis for conversations
- `/api/analysis/intent/`: Get intent recognition results
- `/api/analysis/topics/`: Get topic modeling results
- `/api/recommendations/`: Get real-time recommendations for agents

## Contributing

We welcome contributions to ConvoInsight! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to contribute to the project.

## Datasets

The project utilizes the following datasets for training and testing:

1. Relational Strategies in Customer Interactions (RSiCS)
2. 3K Conversations Dataset for ChatBot from Kaggle
3. Customer Support on Twitter Dataset from Kaggle

For more details on these datasets, please refer to the project documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project is part of the "Advanced Certification Course on Computational Data Science at [IISC](https://iisc.ac.in/)" with [TalentSpirit](https://talentsprint.com/course/computational-data-science-iisc-bangalore).
- We acknowledge the use of open-source libraries and frameworks that made this project possible.

## Disclaimer

This project is intended as a learning exercise and demonstration of integrating various technologies, including:

- Django
- Django Channels
- Django REST Framework
- LangChain
- Hugging Face Endpoint APIs
- Finetuning LLM
- Machine Learning Algorithms for Sentiment analysis, Intent Recognition & Topic Modelling

Please note that this application is not designed or tested for production use. It serves as an educational resource and a showcase of technology integration rather than a production-ready web application.

Contributors and users are welcome to explore, learn from, and build upon this project for educational purposes.