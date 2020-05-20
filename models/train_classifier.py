#-------------------- Import Libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])


#-------------------- Function to change the value of 2 to 1 in a dataframe column
def change_2(x):
    if x == 2:
        return 1
    else:
        return x


#-------------------- Loading the data
def load_data(database_filepath):

    # load data from SQL database
    loc = 'sqlite:///'+database_filepath
    engine = create_engine(loc)
    df = pd.read_sql_table('disasterTable', engine)

    # Apply the 'change_2' function on the related column
    # Since all other columns extracted from the original 'categories' column has either 1's or 0's
    # except for 'related' column, from the extracted 'categories', which has 0's, 1's and 2's.
    # Here I have considered 2's to be the response and converted it to 1's
    df['related'] = df['related'].apply(change_2)

    # Storing the dataframe in X and y. And the categories name in the c_n.
    # We do this to train the model
    X = df['message']
    y = df.iloc[:, 4:]
    c_n = list(df.columns[4:])
    
    return X, y, c_n


#-------------------- Function that uses NLP to convert sentences in tokens
stop_words = stopwords.words("english")

def tokenize(text):

    # We'll remove all the punctuations from a sentence
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Then we'll seperate words
    tokens = word_tokenize(text)

    # Initilizing a lemmatize function 
    lemmatizer = WordNetLemmatizer()
    
    # Using list comprehension to get a list of lemmatized (root words) words which are not in the stopwords   
    token = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
    return token


#-------------------- Function to build a model using Pipeline
def build_model():
    ''' 
    Creating a pipeline function so as to chain multiple estimators into one and hence, 
    automates the machine learning process. 

    Here we also use transformation. 

    The CountVectorizer to convert a collection of text documents to a vector of term/token counts. 
    It also enables the ​pre-processing of text data

    We'll convert this tokens into TF-IDF form by using TfidfTransformer

    The output is used to train the classifier model. We've also used MultioutputClassifier to get multiple results.

    The 'parameter' dictionary is used as a parameter for GridSearchCV to determine and fit the best parameters for the classifier
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__min_samples_split': [2, 4],
              'clf__estimator__n_estimators': [10, 50]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


#-------------------- Function to evaluate the model and print classification report
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test, y_pred_test, target_names=category_names))


#-------------------- Function to save the model to predict the queries from the web app
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()