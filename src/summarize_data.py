from import_data import getData
import matplotlib.pyplot as plot
import numpy as np

def plotSettings():
    pieFigure, (dimensions11, dimensions12) = plot.subplots(1, 2)
    pieFigure.set_figheight(10)
    pieFigure.set_figwidth(20)

    chartFigure, (dimensions21, dimensions22) = plot.subplots(1, 2)
    chartFigure.set_figheight(10)
    chartFigure.set_figwidth(20)

    return (dimensions11, dimensions12, dimensions21, dimensions22)

def summarizeDataset(dataset):
    # shape
    print('\n#### (lin, col) ####')
    print(dataset.shape)
    # head
    print('\n#### Data Examples ####')
    print(dataset.head(10))
    # descriptions
    print('\n#### Descriptions ####')
    print(dataset.describe())

    print('\n#### Average/Mean ####')
    print(dataset.mean(numeric_only=True))

    print('\n#### Standard ####\n')
    print(dataset.std(numeric_only=True))
    print('\n')

    ax11, ax12, ax21, ax22 = plotSettings()
    plot.savefig("output2.png")
    # Average Plot
    dataset.describe().loc['mean'].plot.pie(title='Average/Mean', ax = ax11, startangle = 90)
    dataset.describe().loc['mean'].plot.bar(title='Average/Mean', ax = ax21)
    # Standard Plot
    dataset.describe().loc['std'].plot.pie(title='Standard deviation', ax = ax12, startangle = 90)
    dataset.describe().loc['std'].plot.bar(title='Standard deviation', ax = ax22)
    plot.savefig("../results/mean_and_std.png")
    plot.show()


if __name__ == '__main__':
    dataset = getData()
    summarizeDataset(dataset)