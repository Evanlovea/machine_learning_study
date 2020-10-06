import processData
from sklearn.impute import SimpleImputer
import CombineAttributesAdder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

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
imputer = SimpleImputer(strategy="median")
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
attr_adder = CombineAttributesAdder.CombineAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
print(housing_extra_attribs)