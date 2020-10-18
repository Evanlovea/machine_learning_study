import processData
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class PipelineFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(PipelineFriendlyLabelBinarizer, self).fit_transform(X)


class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedroom_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
#  solve the problem that pandas can not show all the detailed data
# max_columns/max_rows/max_colwidth/line_width
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
stratified_train_data, stratified_test_data = processData.stratified_sampling()
for set in (stratified_train_data, stratified_test_data):
    set.drop(["income_cat"], axis=1, inplace=True)

# copy the data set
housing = stratified_train_data.drop("median_house_value", axis=1)
housing_labels = stratified_train_data["median_house_value"].copy()

# categories for solving NULL value problems
# option 1: give up these areas
# option 2:give up these attributes
# option 3:set the null value into a value
housing.dropna(subset=["total_bedrooms"])  # option 1
housing.drop("total_bedrooms", axis=1)  # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)  # option 3
# method provided by scikit-learn
# create a copy without text attributes
housing_num = housing.drop("ocean_proximity", axis=1)

print(housing_num)
imputer = Imputer(strategy="median")
# use method fit() to adapt imputer into train data
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
# change null value into median value
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
print(housing_tr)

# process text and classify attribute
# change text into number
# encoder = LabelEncoder()
# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# # use classes_ to see the map of the  encoder
# print(encoder.classes_)
# # exchange int_type_value into one_hot_encoder
# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)
# use method LabelBinarizer can implement above two methods
# encoder = LabelBinarizer()
# housing_cat = housing["ocean_proximity"]
# housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)
attr_adder = CombineAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
print(housing_extra_attribs)
print("housing_num")
print(housing_num)

# Conversion pipeline
# list():

num_attribs = list(housing_num)
print("num_attribs")
print(num_attribs)
cat_attribs = ["ocean_proximity"]
print("cat_attribs")
print(cat_attribs)
DFS = DataFrameSelector(num_attribs)

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', PipelineFriendlyLabelBinarizer())
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
# TypeError: fit_transform() takes exactly 2 arguments (3 given)
# Use OneHotEncoder() instead
# Write custom transformer for LabelBinarizer
# Use the older version of sklean which supports your code
# housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_prepared = full_pipeline.fit_transform(housing)
print("housing_prepared")
print(housing_prepared)

