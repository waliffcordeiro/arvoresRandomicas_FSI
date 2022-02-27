def compareResults(results):
    bestResult=[[0], 0, "Aux"]
    for key, value in results.items():
        print("\n{}: Média AUC: {}".format(key, value[0]))
        if value[0] > bestResult[1]:
            bestResult=[key, value[0], value[1]]
    return bestResult
    