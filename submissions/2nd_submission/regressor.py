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

def loss(y_true, y_pred):
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    true = np.maximum(5., np.log10(np.maximum(1., y_true)))
    pred = np.maximum(5., np.log10(np.maximum(1., y_pred)))
    
    loss = np.mean(np.abs(true - pred))
    
    return loss
    
fan_loss = make_scorer(loss, greater_is_better=False)

from sklearn.base import clone

def train_estimators(estimators, X, y):
    # Fitting a list of classifiers
    cpt = 0
    for m in estimators:
        print(cpt)
        m.fit(X, y)
        cpt+=1

def predict_estimators(estimators, X):
    # Get predictions of a list of classifiers
    X_meta = np.zeros((X.shape[0], len(estimators)))
    for m in estimators:
        y_pred = m.predict(X)
        X_meta[:, i] = y_pred
    return X_meta

def predict_ensemble(estimators, clf, X):
    X_meta = predict_estimators(estimators, X)
    return meta_learner.predict(X_meta)

def stacking(estimators, clf, X, y, gen):
    """Simple training routine for stacking."""
    train_estimators(estimators, X, y)
    print('Starting CV training')
    # Generate predictions for training the final classifier clf
    cv_preds, cv_y = [], []
    for i, (train_idx, test_idx) in enumerate(gen.split(X)):
        print(i)
        fold_xtrain, fold_ytrain = X[train_idx, :], y[train_idx]
        fold_xtest, fold_ytest = X[test_idx, :], y[test_idx]

        fold_estimators = [clone(model) for model in estimators]
        train_estimators(fold_estimators, fold_xtrain, fold_ytrain)
        fold_P_base = predict_estimators(fold_estimators, fold_xtest)
        # Saving predictions and test set
        cv_preds.append(fold_P_base)
        cv_y.append(fold_ytest)
    print('Done') 
    # Get rows in the right order
    cv_preds = np.vstack(cv_preds)
    cv_y = np.hstack(cv_y)
    clf.fit(cv_preds, cv_y)

    return estimators, clf

class Regressor(BaseEstimator):
    
    def __init__(self):
        self._estimators = [
                            #KNeighborsRegressor(n_neighbors=5),
                            RandomForestRegressor(n_estimators=10, max_depth=50),
                            GradientBoostingRegressor(n_estimators=10, max_depth=30),
                            XGBRegressor(n_estimators=9,max_depth=30)
                ]
        #self._clf = XGBRegressor(n_estimators=9,max_depth=30)
        self._clf = XGBRegressor(n_estimators=100,max_depth=5)
        #self._clf = GridSearchCV(XGBRegressor(), self._params, cv=5, scoring=fan_loss,verbose=3)

    def fit(self, X, y):

        self._estimators, self._clf = stacking(self._estimators, clone(self._clf), X, y, KFold(3))
        #print(self._clf.feature_importances_)
        #print(self._clf.best_params_)

    def predict(self, X):
        #return self._clf.predict(P)
        return ensemble_predict(self._estimators, self._clf, X)
