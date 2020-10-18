# -*- coding: utf-8 -*-
# @Time    : 2020/10/7 15:32
# @Author  : Evan
# @Email   : evan@stu.haut.edu.cn
# @File    : train_regression_model.py
# @Software: PyCharm
# @Description: train a regression model
import numpy as np
from sklearn.linear_model import LinearRegression
import data_preparation
from sklearn.metrics import mean_squared_error
# train a linearRegression model
lin_reg = LinearRegression()
housing_prepared = data_preparation.housing_prepared
print("housing_prepared", housing_prepared)
housing_labels = data_preparation.housing_labels
print("housing_labels", housing_labels)
housing = data_preparation.housing
lin_reg.fit(housing_prepared, housing_labels)
# use the first 5 rows data to test this model
some_data = housing.iloc[:5]
print("some data")
print(some_data)
# use the first 5 rows of housing_labels as the test labels
some_labels = housing_labels.iloc[:5]
some_data_prepared = data_preparation.full_pipeline.transform(some_data)
# predict
print("Predictions:\t", lin_reg.predict(some_data_prepared))
# compare
print("Labels:\t\t", list(some_labels))

# use mean_squared_error method to test regression model's RMSE in all the data set
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
