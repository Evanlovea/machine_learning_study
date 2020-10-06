import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import loadData

housing = loadData.load_housing_data()


def show_data():

    # hist figure can display number of instances of a given value range (horizontal axis) (vertical axis)
    housing1 = loadData.load_housing_data()
    housing1.hist(bins=50,  figsize=(20, 15))
    plt.show()


def stratified_sampling():
    # housing = loadData.load_housing_data()
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    # combine all categories greater than 5 into 5
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    # stratified sampling
    # @params
    # n_splits means split data set into train/test groups, default 10
    # test_size and train_size is used to set the proportion of train and test in the train/test pair
    #
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return strat_train_set, strat_test_set


if __name__ == '__main__':
    # show_data()
    stratified_train_set, stratified_test_set = stratified_sampling()
    print(housing["income_cat"].value_counts() / len(housing))
    for set in (stratified_train_set, stratified_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

