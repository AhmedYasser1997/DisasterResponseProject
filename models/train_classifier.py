import pickle
import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db)
    Output:
        X -> a dataframe containing features
        y -> a dataframe containing labels
        category_names -> List of categories names
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('messages', engine)
    X = df.iloc[:, 1].values
    y = df.iloc[:, 3:].values
    category_names = df.iloc[:, 3:].columns.values
    return X, y, category_names

def tokenize(text):
    """
    Tokenize the text function and applies operations like stop words & special characters removal, change text to lowercase and lemmatization
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        lemmed -> List of tokens from the provided text
    """
    # Remove special characters
    no_punct = re.sub('[^A-Za-z0-9]+', ' ', text)
    # Tokenize and change text to lowercase
    tokens = word_tokenize(no_punct.lower())
    # Remove stop words if exist
    no_stop_words = [word for word in tokens if word not in stopwords.words('english')]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in no_stop_words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed


def build_model():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    # ML pipeline consisting of a count vectorizer that makes a matrix of token counts.
    # Then a Tfidf transformer that transforms the count matrix to a normalized tf or tf-idf representation
    # And since our labels are multioutput, we use the multioutput classifier which takes a normal estimator/classifier 
    # which was chosen to be the random forest classifier
    
    pipeline = Pipeline([('count', CountVectorizer(tokenizer=tokenize)),
                    ('tfid', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 50, random_state=0)))])
    
    ## GRID SEARCH 
    # The grid search part will take a couple of hours to fit, so I ran it once and saved it in the pipeline_improved variable
    # to avoid wasting time on running it everytime. To return the pipeline resulted from completing the gridsearch, uncomment the following     # lines and replace pipeline with pipeline_improved
    
#     parameters = {'clf__estimator__max_features':['auto', 'sqrt', 'log2'],
#              'clf__estimator__n_estimators':[50, 60, 75]}

#     cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
#     pipeline_improved = Pipeline([('count', CountVectorizer(tokenizer=tokenize)),
#                     ('tfid', TfidfTransformer()),
#                     ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 75, max_features = 'auto', random_state=0)))])

    # In the other ml notebook in the folder, I also explain the same thing 
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance
    
    Arguments:
        model -> A scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    # Predicting the labels
    y_pred = model.predict(X_test)
    
    # Print a performance report for all columns (metrics include: precision, recall & f1-score
    for i in range(0, 35):
        print(category_names[i])
        print(classification_report(Y_test[:,i], y_pred[:,i]))
    
def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipeline object
        model_filepath -> destination path to save .pkl file
    
    """
#     filename = '{}.sav'.format(model_filepath)
#     pickle.dump(model, open(filename, 'wb'))
    pickle.dump(model, open(model_filepath, "wb"))

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