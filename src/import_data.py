from pandas import read_csv
from sklearn import preprocessing

def encoderData(dataset):
    print("#### Imported Data ####")
    print(dataset.head(5))
    le = preprocessing.LabelEncoder()
    for column_name in dataset.columns:
        if dataset[column_name].dtype == object and column_name=="famhist":
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass
    print("\n#### Famhist refactored data ####")
    print(dataset.head(5))
    return dataset

def getData():
    return encoderData(read_csv("../data/SA_heart.csv"))


if __name__ == '__main__':
    print(getData())
    