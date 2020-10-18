# -*- coding: utf-8 -*-
# @Time    : 2020/10/18 21:37
# @Author  : Evan
# @Email   : evan@stu.haut.edu.cn
# @File    : train_random_forest_regressor.py
# @Software: PyCharm
# @Description: train a random forest regressor model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import data_preparation
import numpy as np
from sklearn.externals import joblib

housing_pred = data_preparation.housing_prepared
print("housing_prepared", housing_pred)
housing_labels = data_preparation.housing_labels
print("housing_labels", housing_labels)
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_pred, housing_labels)
scores = cross_val_score(forest_reg, housing_pred, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
print(forest_rmse_scores)


def display_scores(_scores):
    print("scores:", _scores)
    print("mean", _scores.mean())
    print("Standard deviation:", _scores.std())


if __name__ == '__main__':
    display_scores(forest_rmse_scores)
    joblib.dump(forest_reg, "my_model.pkl")
    # and later
    my_model_loaded = joblib.load("my_model.pkl")
