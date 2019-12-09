from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.model_selection import KFold
from numpy import np

class Regressor(BaseEstimator):
    
    def __init__(self):
        self._clf = XGBRegressor(n_estimators=9,max_depth=30)

    def fit(self, X, y):
        self._clf.fit(X ,y)

    def predict(self, X):
        return self._clf.predict(P)
