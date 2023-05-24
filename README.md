[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/4E0vXSSt)
# Practical Applications in Machine Learning 

# Homework 3: Predicting Product Review Sentiment Using Classification

The goal of this assignment is to build a classification machine learning (ML) pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Using trained classification models, build a ML application that predicts whether a product review is positive or negative.

The <b>learning outcomes</b> for this assignment are:
* Build end-to-end regression pipeline using 1) multiple regression, 2) polynomial regression, and 3) ridge, and 4) lasso regression.
* Evaluate regression methods using standard metrics including root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).
* Develop a web application that walks users through steps of the regression pipeline and provide tools to analyze multiple methods across multiple metrics. 
* Develop a web application that offers a service to customers by predicting housing prices using regression models. 

The <b>learning outcomes</b> for this assignment are:
* Build end-to-end classification pipeline with four classifiers 1) Logistic Regression, 2) Stochastic Gradient Descent, 3) Stochastic Gradient Descent with Cross Validation, and 4) Majority Class.
* Evaluate classification methods using standard metrics including precision, recall, and accuracy, ROC Curves, and area under the curve.
* Develop a web application that walks users through steps of the classification pipeline and provide tools to analyze multiple methods across multiple metrics. 
* Develop a web application that classifies products as positive or negative and indicates the cost of displaying false positives and false negatives using a specified model.

# Assignment Deadline:
* Due:  Monday April 17, 2023 at 11:00PM 
* Assignment Type: Individual
* Time Estimate: 18 Hours
* What to turn in: Submit responses on GitHub AutoGrader and Canvas Reflection Assessment 3
* Join Github Classroom: [TODO: Add invitation link] (11 points)
* Submit Reflection Assessment via Canvas (8 multiple choice questions)

## Assignment Outline
* Setup
* End-to-End Regression Models
* Testing Code with Github Autograder
* Reflection Assessment

# Reading Prerequisite 

* Review the jupyter notebook in Chapter 9 Unsupervised Learning of “Machine Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow.” O’Reilly Media, Inc., 2022. Available on Canvas under ‘Library Reserves’.

# Computing Prerequisites

* [Optional] Revisit the [Preliminaries Github repository](https://github.com/Cornell-Tech-PAML-Course/0-paml-preliminaries) if you have not set up your computing environment. 
* Clone Homework 3 Github using the [Github Autograder Invitation Link](TODO).
* Update conda on your machine. Open a terminal window with your conda environment activated and run:
```
conda update conda
```

# End-to-End Regression Models for Housing Prices 

The goal of this assignment is to train multiple classification algorithms using the Amazon Products dataset, evaluate model performance, and deploy the model in a sentiment predict application. You can assume that all User Interface (UI) elements are done for you. Your goal is to fill in the checkpoint functions that demonstrate your understanding of classification models. 

We have provided helper functions defined in the help_functions.py file. See Appendix at the end of this document on helper functions.

# Amazon Products Dataset

This assignment involves training and evaluating ML end-to-end pipeline in a web application using the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of natural language processing (NLP), information retrieval (IR), and machine learning (ML), amongst others. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews. There are many features, but the <b>important</b> features include:
* name: name of Amazon product	
* reviews.text: text in review	
* reviews.title: title of reviews	

Download the dataset from [Google Drive](https://drive.google.com/file/d/1xflMtl_Q7K9_gimvhoung3Snfb96T-8a/view?usp=sharing) and store it in the datasets directory in the repository.

# Explore and Preprocess Data (3 points)

The goal of this page is to explore and preprocess the dataset. First, import the dataset from your machine. We have provided code to remove unuseful features using the clean_data() helper function (see helper_function.py). Then, remove punctuation from the reviews. We have provided UI and functions to summarize text statistics from reviews, search reviews with a keyword, and remove reviews. At the end of this page, encode documents with word counts and Term Frequency Inverse Document frequency features. See details about the checkpoint functions below.

Some activities require a try and except block to train classification models (in later sections).
```
try:
	# write some code
Except ValueError as err:
	st.write({str(err)}) # Print the error message
```

<b>Checkpoint 1</b>: remove_punctuation(df, feature) 

Perform the following tasks in the remove_punctuation function:
* Create a translator using the string library that creates a one to one mapping of a character to its translation/replacement. 
* Write a for loop that iterates through the feature names, check that strings are in the feature. 
* If the features are strings, use the translator to remove punctuation from the strings. It’s recommended that you use a lambda function. 
* Store the updated dataframe df in st.session_state[‘data’].
Example code:
```
translator = str.maketrans('', '', string.punctuation)
for feature_name in features:
	if(df[feature_name].dtype ==’object’):
		df[feature_name] = … # add code here
```

<b>Checkpoint 2</b>: Words need to be encoded as integers or floating point values to input to a machine learning algorithm. Your task is to perform word frequency encoding in the word_count_encoder function, which takes four inputs: the pandas dataframe (df), a list of the feature(s) to perform work count encoding on the given features (feature) and a list of strings with word encoding names 'TF-IDF', 'Word Count' (word_encoder). The function performs work count encoding on the given features and returns the data frame with word count encoded features (df). Perform the following tasks in the word_count_encoder function:
* Use the CountVectorizer() to create a count vectorizer class object. 
* Use the count vectorizer transform() function to the feature in df to create frequency counts for words.
* Convert the frequency counts to an array using the toarray() function and convert the array to a pandas dataframe.
* Add a prefix to the column names in the data frame created in Step 3 using add_prefix() pandas function with ‘word_count_’ as the prefix.
* Add the word count dataframe to df using the pd.concat() function.
* Update the confirmation statement to show the length of the word_count dataframe.

<b>Checkpoint 3</b>: Next, we want to perform TF-IDF encoding, which quantifies the importance or relevance of words or phrases. Fill in code for the tf_idf_encoder function, which takes three inputs: the pandas dataframe (df), a list of the feature(s) to perform tf-idf encoding on (feature), and a list of strings with word encoding names ‘TF-IDF’, ‘Word Count’ (word_encoder). This function returns the dataframe with TF-IDF encoded feature(s). Perform the following tasks in the tf_idf_word_count_encoder function:
* Use the CountVectorizer() to create a count vectorizer class object. 
* Use the count vectorizer transform() function to the feature in df to create frequency counts for words.
* Use the TfidfTransformer() to create a TF-IDF transformer class object. 
* Transform the frequency counts (from Step 2) into TF-IDF features using the TfidfTransformer object.
* Create a pandas dataframe for the TF-IDF features which takes the TF-IDF features array as input so convert the TF-IDF features to an array using the toarray() function.
* Add a prefix to the column names in the data frame created in Step 3 using add_prefix() pandas function with ‘tf_idf_word_count_’ as the prefix.
* Add the TF-IDF dataframe to df using the pd.concat() function.

# Train Regression Models (6 points)

The goal of this page is to train multiple models and inspect model coefficients and cross validation results for relevant models. First, we have provided code to assign the negative values to the product ratings using the negative ratings selected from the user (assume rating=3 is neural) in the set_pos_neg_reviews() function. Then, write the split_dataset() function which splits the dataset into training and validation input and output using the appropriate word encoding (as specified by the user). Next, write four functions to training functions to train the following models: 1) Logistic Regression, 2) Stochastic Gradient Descent, and 3) Stochastic Gradient Descent with Cross Validation. Lastly, write a function to inspect the coefficients of each model. 

<b>Checkpoint 4</b>: Before training the models, you need to split the data set into training and test sets. Complete the split_dataset function which takes the following inputs: training features (X), training targets (y), the ratio of test samples (number). The function returns the training features
```
(df, test_size, target, feature_encoding, random_state=42) 
```

As input, pass the data matrix X along with the corresponding target vector y into scikit-learn’s train_test_split() function. Set the default test_size to 0.2, and the default random_state to 42. The function will output four objects, in the following order: X_train, X_val, y_train, y_val. Refer to the scikit-learn train_test_split() function for help.

This function splits the dataset into the training and test sets. 
Input: 
* X: Data matrix of independent features
* y: Target column vector 
* test_size: the ratio of test samples 
Output: 
* X_train: training features 
* X_val: validation features 
* y_train: training target 
* y_val: validation target

Perform the following tasks in the split_dataset function:
* Use the train_test_split() function to split the dataset into four parts including X_train, X_val, y_train, y_val sets using the input X, y, number/100 (set test percentage), and random state.
* Check the feature_encoding list of strings that contain either ‘TF-IDF’ or ‘Word Count’ ing the feature_encoding list and set the input to feature names that start with ‘tf_idf_word_count_’ for TF-IDF and ‘word_count_’ for word count (see example below). Also, the dataset can contain both feature encodings.

```
if(‘Word Count’ in feature_encoding):
    X_train_sentiment = X_train.loc[:, X_train.columns.str.startswith('word_count_')] X_val_sentiment = X_val.loc[:, X_val.columns.str.startswith('word_count_')]
```

<b>Checkpoint 5</b>: train_logistic_regression(X_train, y_train, model_name, random_state=42) 

Perform the following tasks in the train_logistic_regression function:
* Create a [try and except](https://pythonbasics.org/try-except/) block to train a logistic regression model.
* Create a LogisticRegression class object using the random_state as input.
* Fit the model to the data using the fit() function with input data X_train, y_train. Remember to create a continuous y_train array using np.ravel() function.
* Save the model in st.session_state[model_name]. 

<b>Checkpoint 6</b>: train_sgd_classifer(X_train, y_train, model_name, params, random_state=42) 

Perform the following tasks in the train_sgd_classifer function:
* Create a try and except block to train a logistic regression model with Stochastic Gradient Descent algorithm.
* Create a SGDClassifier class object using the random_state and params as input.
```
sgd_model = SGDClassifier( random_state=random_state, **params)
```
* Fit the model to the data using the fit() function with input data X_train, y_train. Remember to create a continuous y_train array using np.ravel() function.
* Save the model in st.session_state[model_name]. 

<b>Checkpoint 7</b>: train_sgdcv_classifer(X_train, y_train, model_name, params, random_state=42) 

Perform the following tasks in the train_sgdcv_classifer function:
* Create a try and except block to train a logistic regression model with Stochastic Gradient Descent algorithm with Repeated K-Fold Cross Validation and search for the optimal parameters with gridsearch.
* Create a SGDClassifier class object using the random_state and params as input.
```
sgd_model = SGDClassifier(max_iter=params['max_iter'], 
                            tol=params['tol'], 
                            penalty=params['penalty'][0], 
                            alpha=params['alpha'], 
                            loss=params['loss'], 
                            random_state=random_state)
```
* Create a RepeatedKFold object with random_state and cv_params 
	```
	sgd_cv = RepeatedKFold(n_splits=cv_params['n_splits'], 
                          n_repeats = cv_params['n_repeats'], 
                                               random_state=random_state)
	```
* Define sgd_results variable using cross_validate object with parameters: sgd_model, X_train, y_train, scoring='accuracy', cv=sgd_cv, n_jobs=-1, return_estimator=True
* Define best_model_idx as the index with the minimum score (np.argmin) in sgd_results['test_score']
```
best_model_idx = np.argmin(...)
```
* Save the model in sgd_results['estimator'] with the best score at index best_model_idx
* Save the best model in st.session_state[model_name]. 

<b>Checkpoint 8</b>: train_majority_classifer(X_train, y_train, model_name) 

Perform the following tasks in the train_majority_classifer function:
* Create a try and except block to train a majority-class model. 
* Create a MajorityClassifier class object using the random_state as input. MajorityClassifier is defined in helper_functions.py.
* Fit the model to the data using the fit() function with input data X_train, y_train (do NOT use np.ravel). 
* Save the model in st.session_state[model_name]. 

<b>Checkpoint 9</b>: inspect_coefficients(trained_models) 

Perform the following tasks in the inspect_coefficients function:
* Write a for loop through the model names and trained models.
```
for name, model in trained_models.items():
```
* In the food loop, 
    * check that the model is not None
    * If the model is valid, store the coefficients in out_dict[name] using model.coef (same for all models)
    * Compute the number of positive and negative coefficients and print the results.
* Save the model in st.session_state[model_name]. 

# Test Regression Models (2 points)

The goal of this page is to evaluate the classification models using precision, recall, accuracy, and ROC Curves. First, the user selects the performance metrics for evaluation. Then, they select the classification models to evaluate. Using the aforementioned inputs, two functions, 1) that computes the evaluation metrics using a trained model and 2) displays a ROC Curve using precision and recall. At the end of this page, the user can select a model to deploy on the ‘Deploy App’ page.

<b>Checkpoint 10</b>: compute_eval_metrics(X, y_true, model, metrics) 

Perform the following tasks in the compute_eval_metrics function:
* Make a prediction using the model and input data
* Write a for loop that iterates through metrics, a list containings one or more strings including ‘precision’, ‘recall’, ‘accuracy’
* Check the metric name and compute it based on the string input. For example, if metric=’precision’ then compute the precision on the predicted and input y_true.
* Store the result in out_dict[metric_name]

<b>Checkpoint 11</b>: plot_roc_curve(X_train, X_val, y_train, y_true, trained_model, metric_select, model) 

Perform the following tasks in the plot_roc_curve function:
* Make a predicting using the trained_model on the input training dataset X_train
* Make a predicting using the trained_model on the input validation dataset X_val
* Plot two ROC Curves using RocCurveDisplay function using RocCurveDisplay.from_predictions (see example below) with predictions on 
    * training data (figure 1) 
    * validation data (figure 2)
    ```
        train_display = RocCurveDisplay.from_predictions(y_train, y_train_pred)
    ```
* Return the RocCurveDisplay predictions on training and test sets

# Deploy ML Application (1 point)

The goal of this page is to deploy a Product Sentiment Classification application that takes a user’s review text as input and predicts whether the review is positive or negative. Your goal is to restore the dataset from the previous page. Then, write the deploy_model() function which uses the selected model from Page C and the input text from the user to predict the review sentiment.

<b>Checkpoint 12</b>: deploy_model(df) 

$ Testing Code with Github Autograder

Test your homework solution as needed using Github Autograder. Clone your personal private copy of the homework assignment. Once the code is downloaded and stored on your computer in your directory of choice, you can start testing the assignment. To test your code, open a terminal and navigate to the directory of the homework3.py file. Then, type ‘pytest -v’ and press enter. The ‘-v’ stands for verbose which shows a summary of the passing and failing test cases.
```
pytest -v 
```

The autograder with print feedback to inform you what checkpoint functions are failing. Test homework3.py using this command in your terminal:
```
streamlit run homework3.py 
```

# Reflection Assessment

Complete on Gradescope.





