from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import append, zeros
from matplotlib import pyplot as plot

from import_data import getData
from summarize_data import summarizeDataset
from tree_training import treeExecute
from compare import compareResults

def main():
    dataset = getData()
    summarizeDataset(dataset)

    # Eliminando a coluna que queremos 'prever'
    features = dataset.drop("chd", axis=1).drop("id", axis=1) # Axis = 1 para eliminar coluna e não só a linha
    x = features.values 
    y = dataset['chd'].values

    # Modelo CART 
    decisionTree = DecisionTreeClassifier()
    # Random Forest (m=9)
    randomForest = RandomForestClassifier(n_estimators=100)
    # Random Forest Sqrt (m=3)
    randomForestSQRT = RandomForestClassifier(n_estimators=100, max_features="sqrt")
    
    results={}
    
    results["CART"] = treeExecute(decisionTree, x, y, "CART")
    results["RandomForest"] = treeExecute(randomForest, x, y, "Random Forest")
    results["RandomForestSQRT"] = treeExecute(randomForestSQRT, x, y, "Random Forest Sqrt")

    bestResult = compareResults(results)
    print("\n\nO melhor modelo foi {} com a média AUC: {}\n".format(bestResult[0], bestResult[1]))

    features_size = len(features.columns.values)
    featuresResults = zeros(features_size)

    idx = 0
    for result in bestResult[2]:
        featuresResults[idx] = result
        idx += 1

    fig = plot.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plot.bar(features.columns.values, (featuresResults/features_size))
    plot.title("Nível de importância de cada feature no modelo {}".format(bestResult[0]))
    plot.savefig("../results/features_results.png")
    plot.show()
    

if __name__ == '__main__':
    main()