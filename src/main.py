from import_data import getData
from summarize_data import summarizeDataset
from decision_tree import decisionTree
from random_forest import randomForest

def main():
    dataset = getData()
    summarizeDataset(dataset)

    # Eliminando a coluna que queremos 'prever'
    x = dataset.drop("chd", axis=1).values # Axis = 1 para eliminar coluna e não só a linha
    y = dataset['chd'].values

    decisionTree(x, y)
    randomForest(x, y)

    

if __name__ == '__main__':
    main()