import hashlib
import numpy as np
import loadData

# use a identifier to decide whether it can enter into the test data set
# function digest() returns a binary string value
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio


# split the test data set and the train data set
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# use row index as the ID
# if use this, you should promise that you can only add data at the end of the data set
def use_row_id_split_data(data):
    housing_with_id = data.reset_index()
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    return train_set, test_set


if __name__ == '__main__':
    housing = loadData.load_housing_data()
    train_set, test_set = use_row_id_split_data(housing)


