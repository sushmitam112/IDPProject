from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
'''
This file contains all the code for the machine learning
question in the project. Used ChatGPT to figure out the paramaters
for the different ML models and as a template for how to write them.
'''


def race_selectivity_data_prep(df_race):
    '''
    Preps and splits up the data for the classifier machine learning
    the input,output, training, and testing data.
    '''
    df_Y = df_race.loc[:, ['selective', 'non_selective', 'more_selective']]
    # reverese one-hot encodes the three columns to get one string per institution
    df_Y = df_Y.idxmax(axis=1)
    df_X = df_race.loc[:, [
        'col_white', 'col_black', 'col_asian', 'col_hispa', 'col_pacis',
        'col_amind', 'col_twora']]
    X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                        df_Y,
                                                        test_size=0.33,
                                                        random_state=50)
    return X_train, X_test, y_train, y_test


def run_race_selectivity_ml_models(X_train, X_test, y_train, y_test):
    '''
    Prints out the training and testing accuracies for all three types of ML models used.
    '''
    print(
        'Predicting Selectivity of College From Racial Make-Up of School Enrollment'
    )
    print('Decision Tree Model Test Accuracy:',
          race_selectivity_decision_tree(X_train, X_test, y_train, y_test)[0])
    print('Decision Tree Model Train Accuracy:',
          race_selectivity_decision_tree(X_train, X_test, y_train, y_test)[1])
    print('Random Forest Test Accuracy:',
          race_selectivity_random_forest(X_train, X_test, y_train, y_test)[0])
    print('Random Forest Train Accuracy:',
          race_selectivity_random_forest(X_train, X_test, y_train, y_test)[1])
    print('K-Nearest Neighbors Test Accuracy:',
          race_selectivity_knn(X_train, X_test, y_train, y_test)[0])
    print('K-Nearest Neighbors Train Accuracy:',
          race_selectivity_knn(X_train, X_test, y_train, y_test)[1])


def race_selectivity_decision_tree(X_train, X_test, y_train, y_test):
    '''
    Decision Tree Classifier - returns training/testing accuracy
    '''
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_predict = dt_model.predict(X_test)
    y_predict_train = dt_model.predict(X_train)
    return [
        accuracy_score(y_test, y_predict),
        accuracy_score(y_train, y_predict_train)
    ]


def race_selectivity_random_forest(X_train, X_test, y_train, y_test):
    '''
    Random Forest Classifier - returns training/testing accuracy
    '''
    rf_model = RandomForestClassifier(n_estimators=200, random_state=70)
    rf_model.fit(X_train, y_train)
    y_predict = rf_model.predict(X_test)
    y_predict_train = rf_model.predict(X_train)
    return [
        accuracy_score(y_test, y_predict),
        accuracy_score(y_train, y_predict_train)
    ]


def race_selectivity_knn(X_train, X_test, y_train, y_test):
    '''
    KNN Classifer - returns training/testing accuracy
    '''
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train, y_train)
    y_predict = knn_model.predict(X_test)
    y_predict_train = knn_model.predict(X_train)
    return [
        accuracy_score(y_test, y_predict),
        accuracy_score(y_train, y_predict_train)
    ]


def socio_and_SAT_data_prep(df_merged):
    '''
    Preps and splits up the data for the regression machine learning
    data for input,output, training, and testing data.
    '''
    # filters the needed input columns
    df_filtered = df_merged[[
        'FAMINC', 'MD_FAMINC', 'FIRST_GEN', 'SAT_AVG', 'UGDS_WHITE',
        'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN', 'UGDS_AIAN', 'UGDS_NHPI'
    ]]
    
    # the following code was needed for the pre-filtered recent cohorts file
    df_filtered = df_filtered.dropna()
    for col in df_filtered.columns:
        df_filtered = df_filtered[~(df_filtered[col] == 'PrivacySuppressed')]
    df_Y = df_filtered['SAT_AVG']
    # takes output column from input dataframe
    df_X = df_filtered.drop(columns=['SAT_AVG'])
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.33,
                                                        random_state=50)
    return X_train, X_test, y_train, y_test


def run_socio_and_SAT_data_ml_models(X_train, X_test, y_train, y_test):
    '''
    Prints out the training and testing accuracies for all three types of ML models used.
    '''
    print('Predicting Average SAT Score of College From Socio-economic Make-Up of School ')
    print('Decision Tree Model Test Error:',
          socio_SAT_decision_tree(X_train, X_test, y_train, y_test)[0])
    print('Decision Tree Model Train Error:',
          socio_SAT_decision_tree(X_train, X_test, y_train, y_test)[1])
    print('Random Forest Test Error:',
          socio_SAT_random_forest(X_train, X_test, y_train, y_test)[0])
    print('Random Forest Train Error:',
          socio_SAT_random_forest(X_train, X_test, y_train, y_test)[1])
    print('SVM Test Error:',
          socio_SAT_svm(X_train, X_test, y_train, y_test)[0])
    print('SVM Neighbors Train Error:',
          socio_SAT_svm(X_train, X_test, y_train, y_test)[1])


def socio_SAT_decision_tree(X_train, X_test, y_train, y_test):
    '''
    Returns the error for the Decision Tree Regression model.
    '''
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    y_predict_train = dt.predict(X_train)
    score1 = mean_squared_error(y_test, y_predict, squared=False)
    score2 = mean_squared_error(y_train, y_predict_train, squared=False)
    return [score1, score2]


def socio_SAT_random_forest(X_train, X_test, y_train, y_test):
    '''
    Returns the error for the Random Forest Regression model.
    '''
    rf_model = RandomForestRegressor(n_estimators=100, random_state=50)
    rf_model.fit(X_train, y_train)
    y_predict = rf_model.predict(X_test)
    y_predict_train = rf_model.predict(X_train)
    score1 = mean_squared_error(y_test, y_predict, squared=False)
    score2 = mean_squared_error(y_train, y_predict_train, squared=False)
    return [score1, score2]


def socio_SAT_svm(X_train, X_test, y_train, y_test):
    '''
    Returns the error for the Support Vector Machine model.
    '''
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    y_predict = svm_model.predict(X_test)
    y_predict_train = svm_model.predict(X_train)
    score1 = mean_squared_error(y_test, y_predict, squared=False)
    score2 = mean_squared_error(y_train, y_predict_train, squared=False)
    return [score1, score2]
