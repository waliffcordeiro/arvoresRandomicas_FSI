from import_data import getData
from summarize_data import summarizeDataset
from decision_tree import decisionTree

def main():
    dataset = getData()
    summarizeDataset(dataset)
    decisionTree(dataset)

    

if __name__ == '__main__':
    main()