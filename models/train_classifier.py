# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.model_selection import GridSearchCV

import pickle


import nltk
nltk.download(['punkt', 'wordnet','stopwords'])


def load_data(database_filepath):
    """
    Funtion to load data from database
    Parameter: 
        database_filepath: path of the database 
    Returns:
        - X: features (messages)
        - Y: Labels (categories)
        - categories_names: columns of the labels
    """ 
    engine = create_engine('sqlite:///{:}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponseMessages", con=engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    """ 
    Function to transform text in tokens
    parameters:
        text: text to be transformed
    Return:
        list of tokens
    """
    
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function to build a model for machine learning
    return:
        model for machine learning
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators= 50, random_state=42)))
    ])
    
    parameters =  {'clf__estimator__n_estimators': [50,100], 'clf__estimator__max_features': ['auto','sqrt']} 

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate a model
    Parameters:
        model: The model to be evaluated
        X_test: features for testing
        Y_test: labels for testing
        category_names: columns of the labels
    """
    
    Y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):

            accuracy=accuracy_score(Y_test.loc[:,col],Y_pred[:,i])
            print("[{:}] - accuracy: {:.2f}\n".format(col,accuracy))
            print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Function to save a model
    Parameters:
        model: model to be saved
        model_filepath: file path where the model will be saved
    """
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train) # model is GridSearchCV on the ML pipeline
        
        print("Output the parameters of the underlying random forest classifier...")
        print(model.best_params_) # output the parameters of the underlying random forest classifier
        model = model.best_estimator_ # After training the model is set to the "best_estimator_" of the GridSearchCV
        
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