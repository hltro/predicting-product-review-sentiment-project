import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions import fetch_dataset, compute_precision, compute_recall, compute_accuracy, apply_threshold
from sklearn.metrics import recall_score, precision_score, accuracy_score
from pages.B_Train_Model import split_dataset

random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown(
    "### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.title('Test Model')

#############################################

# Used to access model performance in dictionaries
METRICS_MAP = {
    'precision': compute_precision,
    'recall': compute_recall,
    'accuracy': compute_accuracy
}

# Checkpoint 9
def compute_eval_metrics(X, y_true, model, metrics):
    """
    This function computes one or more metrics (precision, recall, accuracy) using the model

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {'precision': -1,
                   'recall': -1,
                   'accuracy': -1}
    # Add code here
    y_pred = model.predict(X)
    
    for metric_name in metrics:
        if metric_name == 'precision':
            # Compute precision using precision_score
            precision = precision_score(y_true, y_pred)
            metric_dict['precision'] = precision
        elif metric_name == 'recall':
            # Compute recall using recall_score
            recall = recall_score(y_true, y_pred)
            metric_dict['recall'] = recall
        elif metric_name == 'accuracy':
            # Compute accuracy using accuracy_score
            accuracy = accuracy_score(y_true, y_pred)
            metric_dict['accuracy'] = accuracy    
    
    return metric_dict

# Checkpoint 10
def plot_roc_curve(X_train, X_val, y_train, y_val, trained_models, model_names):
    """
    Plot the ROC curve between predicted and actual values for model names in trained_models on the training and validation datasets

    Input:
        - X_train: training input data
        - X_val: test input data
        - y_true: true targets
        - y_pred: predicted targets
        - trained_model_names: trained model names
        - trained_models: trained models in a dictionary (accessed with model name)
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[model_name.__name__ + " Train Precision"] = train_precision_all
            - df[model_name.__name__ + " Train Recall"] = train_recall_all
            - df[model_name.__name__ + " Validation Precision"] = val_precision_all
            - df[model_name.__name__ + " Validation Recall"] = val_recall_all
    """
    # Set up figures
    fig = make_subplots(rows=len(trained_models), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)

    # Intialize variables
    df = pd.DataFrame()
    threshold_values = np.linspace(0.5, 1, num=100)
    
    # Write a for loop that iterates through the trained model names with an enumerator (e.g., i) variable to use for plotting
    for i, model_name in enumerate(model_names):
        # Make predictions on the train set using predict_proba() function
        model = trained_models[i]
        train_precision_all, train_recall_all, val_precision_all, val_recall_all = [],[],[],[]
        
        for threshold in threshold_values:
            train_preds = model.predict_proba(X_train)
            val_preds = model.predict_proba(X_val)
            # Apply threshold to train set & validation set using apply_threshold()
            train_preds = apply_threshold(train_preds, threshold)
            val_preds = apply_threshold(val_preds, threshold)
            
            # Compute precision and recall on train set using precision_score and recall_score
            train_precision = precision_score(y_train, train_preds, zero_division=1)
            train_recall = recall_score(y_train, train_preds, zero_division=1)
            
            # Compute precision and recall on validation set using precision_score and recall_score
            val_precision = precision_score(y_val, val_preds, zero_division=1)
            val_recall = recall_score(y_val, val_preds, zero_division=1)
            
            # Store values for plotting
            train_precision_all.append(train_precision)
            train_recall_all.append(train_recall)
            val_precision_all.append(val_precision)
            val_recall_all.append(val_recall)
        
        # print("train_precision_all", len(train_precision_all)) 
        # Add precision and recall values to dataframe
        df[model_name + " Train Precision"] = train_precision_all
        df[model_name + " Train Recall"] = train_recall_all
        df[model_name + " Validation Precision"] = val_precision_all
        df[model_name + " Validation Recall"] = val_recall_all
        
        fig.add_trace(go.Scatter(x=train_recall_all, y=train_precision_all, name="Train"), row=i+1, col=1) # use enumerated value i to align figures vertically
        fig.add_trace(go.Scatter(x=val_recall_all, y=val_precision_all, name="Validation"), row=i+1, col=1) # use enumerated value i
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text='Precision', row=i+1, col=1) # use enumerated value i fig.update_layout(title=model_name+' ROC Curve')

    return fig, df


# Page C
def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number, feature_select, 'TF-IDF')
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data_splits(df)

    st.markdown("## Get Performance Metrics")
    metric_options = ['precision', 'recall', 'accuracy']

    classification_methods_options = ['Logistic Regression',
                                      'Stochastic Gradient Descent with Logistic Regression',
                                      'Stochastic Gradient Descent with Cross Validation']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained classification models for evaluation',
        options=trained_models
    )
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('## Review Classification Model Performance')

            plot_options = ['ROC Curve', 'Metric Results']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=plot_options
            )
            ############## Task 10: Compute evaluation metrics
            if 'ROC Curve' in review_plot:
                trained_select = [st.session_state[model]
                                  for model in model_select]
                fig, df = plot_roc_curve(
                    X_train, X_val, y_train, y_val, trained_select, model_select)
                st.plotly_chart(fig)

            ############## Task 11: Plot ROC Curves
            if 'Metric Results' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}

                # Select multiple metrics for evaluation
                metric_select = st.multiselect(
                    label='Select metrics for classification model evaluation',
                    options=metric_options,
                )
                if (metric_select):
                    st.session_state['metric_select'] = metric_select
                    st.write(
                        'You selected the following metrics: {}'.format(metric_select))

                    for idx, model in enumerate(models):
                        train_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_train, y_train, model, metric_select)
                        val_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_val, y_val, model, metric_select)

                    st.markdown('### Predictions on the training dataset')
                    st.dataframe(train_result_dict)

                    st.markdown('### Predictions on the validation dataset')
                    st.dataframe(val_result_dict)

    # Select a model to deploy from the trained models
    st.markdown("## Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=st.session_state['trained_models'],
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to Deploy Model')
