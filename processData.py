import matplotlib.pyplot as plt
import loadData


def show_data():

    # hist figure can display number of instances of a given value range (horizontal axis) (vertical axis)
    housing1 = loadData.load_housing_data()
    housing1.hist(bins=50,  figsize=(20, 15))
    plt.show()


if __name__ == '__main__':
    show_data()
