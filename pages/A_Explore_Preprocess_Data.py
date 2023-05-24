import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from helper_functions import fetch_dataset, clean_data, summarize_review_data, display_review_keyword, remove_review

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.markdown('# Explore & Preprocess Dataset')

# Load the dataset
df = pd.read_csv("./datasets/Amazon Product Reviews I.csv")

#############################################

# Checkpoint 1
def remove_punctuation(df, features):
    """
    This function removes punctuation from features (i.e., product reviews)

    Input: 
        - df: the pandas dataframe
        - feature: the features to remove punctation
    Output: 
        - df: dataframe with updated feature with removed punctuation
    """
    # check if the feature contains string or not
    # Add code here
    translator = str.maketrans('', '', string.punctuation)
    
    # applying translate method eliminating punctuations
    # Add code here
    for feature_name in features:
        if df[feature_name].dtype == 'object':
            df[feature_name] = df[feature_name].apply(lambda x: x.translate(translator))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Confirmation statement
    st.write('Punctuation was removed from {}'.format(features))
    return df

# Checkpoint 2
def word_count_encoder(df, feature, word_encoder):
    """
    This function performs word count encoding on feature in the dataframe

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform word count encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with word count feature
    """
    # Add code here
    # Use the CountVectorizer() to create a count vectorizer class object.
    count_vect = CountVectorizer()
    # Use the count vectorizer transform() function to the feature in df to create frequency counts for words.
    df_counts = count_vect.fit_transform(df[feature])
    # Convert the frequency counts to an array using the toarray() function and convert the array to a pandas dataframe.
    # tfidf_transformer = TfidfTransformer()
    # df_tfidf = tfidf_transformer.fit_transform(df_counts)
    df_counts = pd.DataFrame(df_counts.toarray())
    # Add a prefix to the column names in the data frame created in Step 3 using add_prefix() pandas function with ‘word_count_’ as the prefix.
    word_count_df = df_counts.add_prefix('word_count_')
    # Add the word count dataframe to df using the pd.concat() function.
    df = pd.concat([df, word_count_df], axis=1)
    
    # (Uncomment code) Show confirmation statement
    st.write('Feature {} has been word count encoded from {} reviews.'.format(feature, len(word_count_df)))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Save variables for restoring state
    word_encoder.append('Word Count')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect

    return df

# Checkpoint 3
def tf_idf_encoder(df, feature, word_encoder):
    """
    This function performs tf-idf encoding on the given features

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform tf-idf encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with tf-idf encoded feature
    """
    # Add code here
    # Use the CountVectorizer() to create a count vectorizer class object.
    count_vect = CountVectorizer()
    # Use the count vectorizer transform() function to the feature in df to create frequency counts for words.
    df_counts = count_vect.fit_transform(df[feature])
    # Convert the frequency counts to an array using the toarray() function and convert the array to a pandas dataframe.
    tfidf_transformer = TfidfTransformer()
    df_tfidf = tfidf_transformer.fit_transform(df_counts)
    df_counts = pd.DataFrame(df_tfidf.toarray())
    # Add a prefix to the column names in the data frame created in Step 3 using add_prefix() pandas function with ‘word_count_’ as the prefix.
    word_count_df = df_counts.add_prefix('tf_idf_word_count_')
    # Add the word count dataframe to df using the pd.concat() function.
    df = pd.concat([df, word_count_df], axis=1)
    
    # (Uncomment code) Show confirmation statement
    st.write(
       'Feature {} has been TF-IDF encoded from {} reviews.'.format(feature, len(word_count_df)))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Save variables for restoring state
    word_encoder.append('TF-IDF')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect
    st.session_state['tfidf_transformer'] = tfidf_transformer
    return df

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. See the unprocesses dataset below.')

    st.dataframe(df)

    # Remove irrelevant features
    df, data_cleaned = clean_data(df)
    if (data_cleaned):
        st.markdown('The dataset has been cleaned. Your welcome!')

    ############## Task 1: Remove Punctation
    st.markdown('### Remove punctuation from features')
    removed_p_features = st.multiselect(
        'Select features to remove punctuation',
        df.columns,
    )
    if (removed_p_features):
        df = remove_punctuation(df, removed_p_features)
        # Display updated dataframe
        st.dataframe(df)
        st.write('Punctuation was removed from {}'.format(removed_p_features))

    # Summarize reviews
    st.markdown('### Summarize Reviews')
    object_columns = df.select_dtypes(include=['object']).columns
    summarize_reviews = st.selectbox(
        'Select the reviews from the dataset',
        object_columns,
    )
    if(summarize_reviews):
        # Show summary of reviews
        summary = summarize_review_data(df, summarize_reviews)

    # Inspect Reviews
    st.markdown('### Inspect Reviews')

    review_keyword = st.text_input(
        "Enter a keyword to search in reviews",
        key="review_keyword",
    )

    # Display dataset
    st.dataframe(df)

    if (review_keyword):
        displaying_review = display_review_keyword(df, review_keyword)
        st.write(displaying_review)

    # Remove Reviews: number_input for index of review to remove
    st.markdown('### Remove Irrelevant/Useless Reviews')
    review_idx = st.number_input(
        label='Enter review index',
        min_value=0,
        max_value=len(df),
        value=0,
        step=1)

    if (review_idx):
        df = remove_review(df, review_idx)
        st.write('Review at index {} has been removed'.format(review_idx))

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)
    word_encoder = []

    word_count_col, tf_idf_col = st.columns(2)

    ############## Task 2: Perform Word Count Encoding
    with (word_count_col):
        text_feature_select_int = st.selectbox(
            'Select text features for encoding word count',
            string_columns,
        )
        if (text_feature_select_int and st.button('Word Count Encoder')):
            df = word_count_encoder(df, text_feature_select_int, word_encoder)

    ############## Task 3: Perform TF-IDF Encoding
    with (tf_idf_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for encoding TF-IDF',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('TF-IDF Encoder')):
            df = tf_idf_encoder(df, text_feature_select_onehot, word_encoder)

    # Show updated dataset
    if (text_feature_select_int or text_feature_select_onehot):
        st.write(df)

    # Save dataset in session_state
    st.session_state['data'] = df

    st.write('Continue to Train Model')
