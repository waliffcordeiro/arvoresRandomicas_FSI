from import_data import getData
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold as KFold
import matplotlib.pyplot as plot
from treeTrainData import treeData

def decisionTree(dataset):

    # Inicializando lists
    aucList = []
    confusionList = []
    legendsList = []

    # Eliminando a coluna que queremos 'prever'
    x = dataset.drop("chd", axis=1).values # Axis = 1 para eliminar coluna e não só a linha
    y = dataset['chd'].values

    # Plot settings
    fig = plot.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)

    KFolds = KFold(n_splits=10)
    tree = DecisionTreeClassifier() # Modelo CART
    numFolds = 0
    for idxTrain, idxTest in KFolds.split(x, y):
        # Contando os folds para plotar os gráficos
        numFolds += 1
        
        # Separando dados de treino e teste
        trainX, trainY, testX, testY = x[idxTrain], y[idxTrain], x[idxTest], y[idxTest]

        # Treinamento
        tree.fit(trainX, trainY)

        # Obtendo dados para cada fold e adicionando nas lists
        aucList, confusionList, legendsList = treeData(tree, testX, testY, numFolds, aucList, confusionList, legendsList)
    plot.legend(legendsList)
    plot.savefig("../results/decisionTree.png")
    plot.show()

    return 

if __name__ == '__main__':
    dataset = getData()
    decisionTree(dataset)