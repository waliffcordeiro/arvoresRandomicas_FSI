# Trabalho da disciplina Fundamentos de Sistemas Inteligentes
Projeto de árvores randômicas para diagnósticos. O arquivo contendo a especificação do projeto está na root no arquivo proj1-fsi-2021_2.pdf
____

Para instalar as dependências necessários você pode rodar o comando 
```
pip install -r requirements.txt
```
Na pasta root deste projeto
____

#### Para executar a solução total do projeto deve-se estar na pasta src e executar o comando
```
python3 main.py
```
___
Alguns outros módulos também possuem suas próprias "versões main" para serem executados e testados individualmente. Isso é feito através do
```
if __name__ == '__main__':
```
____
### Organização do projeto
A função treeExecute é responsável por fazer a execução da lógica por trás do treinamento e obtenção de resultados, sendo que foi projetada para suportar os três modelos (árvore de decisão, Random Forest e Random Forest Sqrt). A utilização dela fica, por exemplo:
```
results["CART"] = treeExecute(decisionTree, x, y, "CART")
results["RandomForest"] = treeExecute(randomForest, x, y, "Random Forest")
results["RandomForestSQRT"] = treeExecute(randomForestSQRT, x, y, "Random Forest Sqrt")
```
Outra função muito importante é a treeTraining que também é utilizada nas três situações de modelo e serve para retornar a AUC média e lista de performance das features.

O código foi desenvolvido de forma modular e comentado, de forma que as instruções necessárias para entendimento estão descritas no decorrer da própria implementação.
____
### Output
Ao executar o código temos como retorno alguns outputs através do terminal, alguns plots de gráficos mas também salva-se as informações mais importantes no folder results, informações como: média e desvio padrão do dataset, matrizes de confusão, importâncias da melhor feature e gráficos de ROC e AUC ROC para cada modelo executado.

O fit das árevores é feito através do arquivo tree_training na função treeExecute que recebe a árvore, x, y e o modelo a ser executado. O arquivo treeTrainData nos auxilia retornando os dados AUC, performance das features e legendas a serem utilizadas no plot dos gráficos

____
### Avaliação de resultados
Não considerei excelente o resultado encontrado, sendo o melhor deles menor que 0.8

Os resultados utilizando apenas o predict estavam muito similares ao da árvore de decisão, encontrei o predict_proba que etorna a precisão média nos dados e rótulos de teste fornecidos, sendo que o gráfico ficou mais estranho, tendo um formato mais "quadrado", cheio de pontas, porém pelo aumento da área teve-se um melhor resultado também.

Apenas com predict o resultado estava oscilando muito entre forestRandom e a Sqrt e os valores muito próximos da decisionTree (nem chegavam em 0.7). Com o uso do predict_proba teve-se uma melhora considerável, apesar de uma estranheza no gráfico
____