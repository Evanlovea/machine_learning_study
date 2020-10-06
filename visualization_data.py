from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import processData


# create a copy from the train data set
stratified_train_data, stratified_test_data = processData.stratified_sampling()
housing = stratified_train_data.copy()
print(housing)
# visualize the geographic data
# set alpha=1, can see location of high-density data points more clearly
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
# use a pre-define color table named jet
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=housing["population"]/100, label="population",c="median_house_value",cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
# plt.show()
# find the correlation
corr_matrix = housing.corr()
# now let us see the correlation between every attribute and median house value
print(corr_matrix["median_house_value"].sort_values(ascending=False))
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()
# the most useful attribute is th median_income
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()