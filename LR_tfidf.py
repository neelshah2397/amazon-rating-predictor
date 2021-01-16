import os, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

def data_reader(path = None):
    """""""""""""""""""""""""""""""""""""""""""""
    Function to read train and test json data
        - path: directory path to the data files
        - return: train & test pandas dataframe
    """""""""""""""""""""""""""""""""""""""""""""
    train_path = os.path.join(path,'train.json')
    test_path = os.path.join(path,'test.json')
    train = pd.read_json(train_path, lines=True)
    test = pd.read_json(test_path, lines=True)
    return train, test


def feature_extraction(train, test):
    """""""""""""""""""""""""""""""""""""""""""""
    Function to extract tf-idf features from review text
        - train and test: training data and test data 
        - return: training feature set with known labels and test feature set
    """""""""""""""""""""""""""""""""""""""""""""
    train = train.fillna('')
    test = test.fillna('')

    vectorizer = TfidfVectorizer(max_features = 3000)
    vectorizer.fit(train['reviewText'].tolist())

    vectors = vectorizer.transform(train['reviewText'])
    X_train = pd.DataFrame(vectors.todense().tolist(),
              columns=vectorizer.get_feature_names())
    y_train = train['overall'].values

    vectors = vectorizer.transform(test['reviewText'])
    X_test = pd.DataFrame(vectors.todense().tolist(),
             columns=vectorizer.get_feature_names())

    return X_train, y_train, X_test


def model_train_test(X_train, y_train, X_test):
    """""""""""""""""""""""""""""""""""""""""""""
    Train model and make predications
        - X_train, y_train and X_test are pandas data frames with the tf-idf features and the correct labels in y_train
        - return: predictions 
    """""""""""""""""""""""""""""""""""""""""""""
    scores = cross_val_score(LinearRegression(), X_train, y_train, cv=4,\
                             n_jobs=-1, scoring='neg_mean_squared_error')
    print("Training Data CV MSE: {}".format(np.mean(np.abs(scores))))

    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    for i in range(len(y_pred)):
        if y_pred[i] > 5 :
            y_pred[i] = 5
        elif y_pred[i] < 0:
            y_pred[i] = 0
            

    return y_pred


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ## Reading the train and test json data
    train, test = data_reader(dir_path)

    ## Processing data to parse text fields and extract features
    X_train, y_train, X_test = feature_extraction(train, test)
    # Training and testing the model for evaluation
    predictions = model_train_test(X_train, y_train, X_test)
    # print final predictions

    result_df = pd.DataFrame(columns=['userID-itemID',	'prediction'])
    result_df['userID-itemID'] = test[['reviewerID', 'itemID']].agg('-'.join,axis=1)
    result_df['prediction'] = predictions
    result_df.to_csv(dir_path + '/results.csv', index=False)
