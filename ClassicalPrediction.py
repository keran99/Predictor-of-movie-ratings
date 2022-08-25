import pandas as pd
import numpy as np

from PreProccessing import preprocessing

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score

def KNeighborsKFold(X_train, X_test, y_train, y_test):
    algorithm = KNeighborsRegressor()
    hp_candidates = [{
        'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        'weights': ['uniform', 'distance']}]
    KFold_val(algorithm, hp_candidates , X_train, X_test, y_train, y_test)

def SVRKFold(X_train, X_test, y_train, y_test):
    algorithm = SVR()
    hp_candidates = [{
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
    KFold_val(algorithm, hp_candidates, X_train, X_test, y_train, y_test)

def DecisionTreeFold(X_train, X_test, y_train, y_test):
    algorithm = DecisionTreeRegressor()
    hp_candidates = [{
        'max_depth': range(4,20,1),
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}]
    KFold_val(algorithm, hp_candidates, X_train, X_test, y_train, y_test)

def RandomForestFold(X_train, X_test, y_train, y_test):
    algorithm = RandomForestRegressor()
    hp_candidates = [{
        'n_estimators': range(100, 1000, 100),
        'max_depth': range(4,20,4),
        'criterion': ['squared_error',  'absolute_error', 'poisson']}]
    KFold_val(algorithm, hp_candidates, X_train, X_test, y_train, y_test)

def gradientBoostingFold(X_train, X_test, y_train, y_test):
    algorithm = GradientBoostingRegressor()
    hp_candidates = {
        'n_estimators': range(1000, 5000, 100),
        'learning_rate': [0.01, 0.1, 0.05]}
    KFold_val(algorithm, hp_candidates, X_train, X_test, y_train, y_test)


def KFold_val(algorithm, hp_candidates , X_train, X_test, y_train, y_test):
    seed = 13
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=algorithm, param_grid=hp_candidates, cv=kfold, scoring='r2')
    grid.fit(X_train, y_train)
    # Get the results
    print(grid.best_score_)
    print(grid.best_estimator_)
    print(grid.best_params_)
    # Evaluate
    predictions = grid.best_estimator_.predict(X_test)
    print("Test:", r2_score(y_test, predictions))

if __name__ == '__main__':
    np.random.seed(123456)
    df = pd.read_csv('ImportDataset.csv')
    X, y, X_train, X_test, y_train, y_test = preprocessing(df, True, 0.02, 10, True, 100, True)
    print('aa')
    print(X)
    KNeighborsKFold(X_train, X_test, y_train, y_test)
    SVRKFold(X_train, X_test, y_train, y_test)
    DecisionTreeFold(X_train, X_test, y_train, y_test)
    RandomForestFold(X_train, X_test, y_train, y_test)
    gradientBoostingFold(X_train, X_test, y_train, y_test)



