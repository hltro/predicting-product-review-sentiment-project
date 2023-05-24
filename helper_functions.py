import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score

# All pages
def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])

        if (data):
            df = pd.read_csv(data)
    if df is not None:
        st.session_state['data'] = df
    return df

# Page A
def clean_data(df):
    """
    This function removes all feature but 'reviews.text', 'reviews.title', and 'reviews.rating'
        - Then, it remove Nan values and resets the dataframe indexes

    Input: 
        - df: the pandas dataframe
    Output: 
        - df: updated dataframe
        - data_cleaned (bool): True if data cleaned; else false
    """
    data_cleaned = False
    # Simplify relevant columns names
    if ('reviews.rating' in df.columns):
        df['rating'] = df['reviews.rating']
        df.drop(['reviews.rating'], axis=1, inplace=True)

    if ('reviews.text' in df.columns):
        df['reviews'] = df['reviews.text']
        df.drop(['reviews.text'], axis=1, inplace=True)

    if ('reviews.title' in df.columns):
        df['title'] = df['reviews.title']
        df.drop(['reviews.title'], axis=1, inplace=True)

    # Drop irrelevant columns
    relevant_cols = ['reviews', 'rating', 'title']
    df = df.loc[:, relevant_cols]

    # Drop Nana
    df.dropna(subset=['rating', 'reviews', 'title'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.head()
    data_cleaned = True

    # Store new features in st.session_state
    st.session_state['data'] = df
    return df, data_cleaned

# Page B
class MajorityClassifier:
    """
    This creates a majority class object with access to the fit() and predict() functions
    """
    def __init__(self):
        self.majority_cls = -1
        self.coef_ = None

    def fit(self, X, y):
        num_positive = int(np.sum(y == +1))
        num_negative = int(np.sum(y == -1))
        self.majority_cls = 1 if num_positive >= num_negative else -1
        return self

    def predict(self, X):
        if self.majority_cls == -1:
            raise Exception('The model has NOT been trained yet.')
        return self.majority_cls

# Page A
def remove_review(X, remove_idx):
    """
    This function drops selected feature(s)

    Input: 
        - X: the pandas dataframe
        - remove_idx: the index of review to be removed
    Output: 
        - X: the updated dataframe
    """

    X = X.drop(index=remove_idx)
    return X

# Page A
def summarize_review_data(df, reviews_col, top_n=3):
    """
    This function summarizes words from reviews in the entire dataset

    Input: 
        - df: the pandas dataframe
        - top_n: top n features with reviews to show, default value is 3
    Output: 
        - out_dict: a dictionary containing the following keys and values: 
            - 'total_num_words': Total number of words
            - 'average_word_count': Average word count per document
            - 'top_n_reviews_most_words': Top n reviews with most words
            - 'top_n_reviews_least_words': Top n reviews with and least words
    """
    out_dict = {'total_num_words': 0,
                'average_word_count': 0,
                'top_n_reviews_most_words': [],
                'top_n_reviews_least_words': []}

    # Compute statistics on reviews and fill in out_dict
    df['Number of Words'] = df[reviews_col].apply(
        lambda sentence: len(sentence.split()))

    total_words = df['Number of Words'].sum()

    out_dict['total_num_words'] = total_words
    out_dict['average_word_count'] = total_words / len(df)
    out_dict['top_n_reviews_most_words'] = df.nlargest(
        top_n, 'Number of Words')#[['reviews']]
    out_dict['top_n_reviews_least_words'] = df.nsmallest(
        top_n, 'Number of Words')#[['reviews']]
    
    # Display summary
    st.write('### Total number of words')
    st.write('Total number of words: {}'.format(out_dict['total_num_words']))

    st.write('### Average word count')
    st.write('Average word count: {}'.format(out_dict['average_word_count']))

    st.write('### N reviews with most words')
    st.dataframe(out_dict['top_n_reviews_most_words'])

    st.write('### N reviews with least words:')
    st.dataframe(out_dict['top_n_reviews_least_words'])

    return out_dict

# Checkpoint 4
def display_review_keyword(df, keyword, n_reviews=5):
    """
    This function shows n_reviews reviews 

    Input: 
        - df: the pandas dataframe
        - keyword: keyword to search in reviews
        - n_reviews: number of review to display
    Output: 
        - None
    """
    keyword_df = df['reviews'].str.contains(keyword)
    filtered_df = df[keyword_df].head(n_reviews)

    return filtered_df

def apply_threshold(probabilities, threshold):
    ### YOUR CODE GOES HERE
    # +1 if >= threshold and -1 otherwise.
    # print(probabilities.shape)
    return np.array([1 if p[1] >= threshold else -1 for p in probabilities])

##################### SOLUTION ONLY FUNCTIONS

def compute_precision(y_true, y_pred):
    """
    Measures the precision between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - precision score
    """
    precision = -1
    precision = precision_score(y_true, y_pred)
    return precision


def compute_recall(y_true, y_pred):
    """
    Measures the recall between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - recall score
    """
    recall = -1
    recall = recall_score(y_true, y_pred)
    return recall


def compute_accuracy(y_true, y_pred):
    """
    Measures the accuracy between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - accuracy score
    """
    accuracy = -1
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Page B
# TODO: won't work if negative_upper_bound==5 --> only has negative y
def set_pos_neg_reviews(df, negative_upper_bound):
    """
    This function updates df with a column called 'sentiment' and sets the positive and negative review sentiment as either -1 or +1

    Input:
        - df: dataframe containing the dataset
        - negative_upper_bound: tuple with upper and lower range of ratings from positive reviews
        - negative_upper_bound: upper bound of negative reviews
    Output:
        - df: dataframe with 'sentiment' column of +1 and -1 for review sentiment
    """
    df = df[df['rating'] != negative_upper_bound]

    # Create a new feature called 'sentiment' and store in df with negative sentiment < up_bound
    df['sentiment'] = df['rating'].apply(
        lambda r: +1 if r > negative_upper_bound else -1)

    # Summarize positibve and negative example counts
    st.write('Number of positive examples: {}'.format(
        len(df[df['sentiment'] == 1])))
    st.write('Number of negative examples: {}'.format(
        len(df[df['sentiment'] == -1])))

    # Save updated df st.session_state
    st.session_state['data'] = df
    return df

# not used
def is_valid_input(input):
    """
    Check if the input string is a valid integer or float.

    Input: 
        - input: string, char, or input from a user
    Output: 
        - True if valid input; otherwise False
    """
    try:
        num = float(input)
        return True
    except ValueError:
        return False