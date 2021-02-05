import pandas as pd
import numpy as np

from decisiontree import DecisionTree
from randomforest import RandomForest
import datasets as d

def test(Datasets, X, Y):
    for dataset, x, y in zip(Datasets, X, Y):
        aDT = []
        aRF = []
        index = []
        print(f"{dataset.name}")
        print()
        print()
        for j in range(d.n):
            print(f"Test {j+1}")
            print()
            trainset = []
            testset = []
            for label, count in dataset[y].value_counts().items():
                samples=dataset[dataset[y]==label]

                testsamples=samples.sample(int((1-d.pTrain)*count))
                testset.append(testsamples)

                #https://stackoverflow.com/questions/28256761/select-pandas-rows-by-excluding-index-number
                trainsamples=samples[~samples.index.isin(testsamples.index)]
                trainset.append(trainsamples)

            Train = pd.concat(trainset).sample(frac=1)
            Test = pd.concat(testset).sample(frac=1)

            print("DECISION TREE - Train")
            DT = DecisionTree()
            DT.train(Train, x, y)


            print("RANDOM FOREST - Train")
            RF = RandomForest(d.num_trees, d.max_samples, d.max_X)
            RF.train(Train, x, y)
            print()

            print("DECISION TREE - Predict")
            aDT.append(accuracy(Test[y], DT.predict(Test)))
            print("RANDOM FOREST - Predict")
            aRF.append(accuracy(Test[y], RF.predict(Test)))

            index.append(f"Test{j+1}")
            print()
            print()

        a = pd.DataFrame(zip(np.around(aDT, 4), np.around(aRF, 4)), columns=["DT - Accuracy", "RF - Accuracy"], index=index)
        a.to_csv(f"{dataset.name}_accuracy.csv")

        mLDT = np.around(np.mean(aDT), 4)
        sLDT = np.around(np.std(aDT), 4)
        mRF = np.around(np.mean(aRF), 4)
        sRF = np.around(np.std(aRF), 4)

        print()
        print("Results:")
        print("DECISIONE TREE - Results:")
        print(f"Mean={mLDT} | STD={sLDT}")
        print()
        print("RANDOM FOREST - Results:")
        print(f"Mean={mRF} | STD={sRF}")

        results = pd.DataFrame([[mLDT, sLDT, mRF, sRF]], columns=["DT - Mean", "DT - STD", "RF - Mean", "RF - STD"])
        results.to_csv(f"{dataset.name}_results.csv", index=False)

        print()
        print()
        print()

def accuracy(y_true, y_pred):
    result = 0
    for a,b in zip(y_true, y_pred):
        if a == b:
            result += 1
    return result/len(y_true)



D, X, Y = d.get_datasets()
test(D, X, Y)
