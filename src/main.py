from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from import_data import getData
from summarize_data import summarizeDataset
from tree_training import treeTraining

def main():
    dataset = getData()
    summarizeDataset(dataset)

    # Eliminando a coluna que queremos 'prever'
    x = dataset.drop("chd", axis=1).values # Axis = 1 para eliminar coluna e não só a linha
    y = dataset['chd'].values

    # Modelo CART 
    decisionTree = DecisionTreeClassifier()
    # Random Forest
    randomForest = RandomForestClassifier(n_estimators=100)
    
    
    treeTraining(decisionTree, x, y, "CART")
    treeTraining(randomForest, x, y, "Random Forest")
    

if __name__ == '__main__':
    main()