from import_data import getData

def summarizeDataset(dataset):
    # shape
    print('\n#### (lin, col) ####')
    print(dataset.shape)
    # head
    print('\n#### Data Examples ####')
    print(dataset.head(20))
    # descriptions
    print('\n#### Descriptions ####')
    print(dataset.describe())

    print('\n#### Average/Mean ####')
    print(dataset.mean(numeric_only=True))

    print('\n#### Standard ####\n')
    print(dataset.std(numeric_only=True))
    print('\n')

if __name__ == '__main__':
    dataset = getData()
    summarizeDataset(dataset)