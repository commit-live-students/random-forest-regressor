from unittest import TestCase
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class TestMyRandomForestRegressor(TestCase):
    def test_myRandomForestRegressor(self):
        from build import finetune_reg
        dataset = loadtxt('./data/boston.csv', delimiter=',', skiprows=1)
        X = dataset[:, 0:8]
        y = dataset[:, 8]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)

        param_grid = {"max_depth": [6, 8, 10],
                      "n_estimators": [10, 20, 30, 40],
                      "max_leaf_nodes": [None, 5, 10, 20],
                      "min_impurity_split": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

        n_iter_search = 5

        y_pred, _ = finetune_reg(X_train, X_test, y_train, param_grid, n_iter_search)

        r2 = r2_score(y_test, y_pred)
        self.assertGreaterEqual(r2, 0.7)