from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from import_data import getData
from summarize_data import summarizeDataset
from tree_training import treeExecute

def main():
    dataset = getData()
    summarizeDataset(dataset)

    # Eliminando a coluna que queremos 'prever'
    x = dataset.drop("chd", axis=1).values # Axis = 1 para eliminar coluna e não só a linha
    y = dataset['chd'].values

    # Modelo CART 
    decisionTree = DecisionTreeClassifier()
    # Random Forest (m=9)
    randomForest = RandomForestClassifier(n_estimators=100)
    # Random Forest Sqrt (m=3)
    randomForestSQRT = RandomForestClassifier(n_estimators=100, max_features="sqrt")
    
    
    treeExecute(decisionTree, x, y, "CART")
    treeExecute(randomForest, x, y, "Random Forest")
    treeExecute(randomForestSQRT, x, y, "Random Forest Sqrt")
    

if __name__ == '__main__':
    main()