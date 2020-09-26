import pandas as pd
import os
import GetData
#  solve the problem that pandas can not show all the detailed data
# max_columns/max_rows/max_colwidth/line_width
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def load_housing_data(housing_path=GetData.HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    housing = load_housing_data()
    print(housing.head())
    # function info() can quickly get the simple description of the data set
    print(housing.info())
    # using function describe() can display the abstract of the values' attributes
    # NAN value will be omitted
    print(housing.describe())
