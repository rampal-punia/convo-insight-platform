# ConvoInsight Prototype

## About
This repository contains a Streamlit-based prototype for ConvoInsight, a Customer Conversational Intelligence Platform. The prototype demonstrates key features of the platform using simulated data and basic analysis techniques.

The `prototype_convochat.py` file is a Streamlit application that simulates the core functionalities of ConvoInsight, including:

- A dashboard with key performance indicators (KPIs) and visualizations
- A chat interface with simulated sentiment analysis, intent detection, and topic extraction
- A conversation analysis page showcasing detailed insights for individual conversations

## Purpose

This prototype serves several purposes:

1. Demonstration: It provides a visual and interactive demonstration of ConvoInsight's potential capabilities.
2. Rapid Iteration: It allows for quick experimentation with different features and user interface designs.
3. Feedback Collection: It can be used to gather early feedback from potential users or stakeholders.
4. Development Guide: It serves as a reference for developers working on the full-scale application.

## Key components include

The prototype uses Streamlit to create a web-based user interface. It employs simulated data and random generation to mimic the behavior of more complex analysis algorithms. Key components include:

- Pandas for data manipulation
- Plotly for interactive visualizations
- Random library for generating simulated data and analysis results

## Running the Prototype Locally

To run this prototype on your local machine, follow these steps:

1. Clone the repository:
   ```
   git clone [repository URL]
   cd [repository name]
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install streamlit pandas plotly
   ```

4. Run the Streamlit app:
   ```
   streamlit run prototype_convochat.py
   ```

5. Open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501)

## Contributing

We welcome contributions to improve this prototype! Here are some ways you can contribute:

- Add new features or visualizations
- Improve the simulated data generation
- Enhance the user interface design
- Optimize performance
- Fix bugs or issues

Please feel free to submit pull requests or open issues to discuss potential improvements.

## Note

This is just a prototype only (Not even the MVP). It uses randomly created simulated data and basic analysis techniques to demonstrate the concept of the ConvoInsight platform.