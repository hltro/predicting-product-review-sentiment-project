import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.markdown("Welcome to the Practical Applications in Machine Learning (PAML) Course! You will build a series of end-to-end ML pipelines, working with various data types and formats, and will need to engineer your system to support training, testing, and deploying ML models.")

st.markdown("""The goal of this assignment is to build a classification machine learning (ML) pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Using trained classification models, build a ML application that predicts whether a product review is positive or negative.

The learning outcomes for this assignment are:
- Build end-to-end classification pipeline with four classifiers 1) Logistic Regression, 2) Stochastic Gradient Descent, 3) Stochastic Gradient Descent with Cross Validation, and 4) Majority Class.
- Evaluate classification methods using standard metrics including precision, recall, and accuracy, ROC Curves, and area under the curve.
- Develop a web application that walks users through steps of the classification pipeline and provide tools to analyze multiple methods across multiple metrics. 
- Develop a web application that classifies products as positive or negative and indicates the cost of displaying false positives and false negatives using a specified model.
""")

st.markdown(""" Amazon Products Dataset

This assignment involves training and evaluating ML end-to-end pipeline in a web application using the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

We have added additional features to the dataset. There are many features, but the important ones include:
- name: name of Amazon product	
- reviews.text: text in review	
- reviews.title: title of reviews	
""")

st.markdown("Click **Explore Dataset** to get started.")