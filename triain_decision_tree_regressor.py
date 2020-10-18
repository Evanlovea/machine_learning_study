# -*- coding: utf-8 -*-
# @Time    : 2020/10/18 20:43
# @Author  : Evan
# @Email   : evan@stu.haut.edu.cn
# @File    : triain_decision_tree_regressor.py
# @Software: PyCharm
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import data_preparation
from sklearn.model_selection import cross_val_score
import numpy as np
# train a decision tree regressor model
tree_reg = DecisionTreeRegressor()
housing_pred = data_preparation.housing_prepared
print("housing_prepared", housing_pred)
housing_labels = data_preparation.housing_labels
print("housing_labels", housing_labels)
# fit this model
tree_reg.fit(housing_pred, housing_labels)
# predict
housing_predictions = tree_reg.predict(housing_pred)
# evaluate the train data set
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Tree RMSE's value is:", tree_rmse)
# There is a problem that the RMSE is 0.0, it shows that the model severely overfits the data
# use cv-participation to better evaluate this model
# k-fold cv-participation
scores = cross_val_score(tree_reg, housing_pred, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
# results


def display_scores(scores):

    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


if __name__ == '__main__':
    display_scores(rmse_scores)

