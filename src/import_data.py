from pandas import read_csv

def getData():
    return read_csv("../data/SA_heart.csv")


if __name__ == '__main__':
    print(getData())