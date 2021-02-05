import pandas as pd

#File modificabile se si vogliono inserire nuovi datasets o cambiare i parametri

pTrain = 0.8 #percentuale di elementi nel trainset
n = 10 #numero di test per dataset

#Parametri per Random Forest
num_trees = 10
max_X = 0.6
max_samples = 0.4

#Funzione per importare datasets
def get_datasets():

    Datasets = []
    Y = []
    X = []

    #http://archive.ics.uci.edu/ml/datasets/Balance+Scale
    ds1 = pd.read_csv("balance-scale.data", names=["Y", "LW", "LD", "RW", "RD"])
    Datasets.append(ds1)
    Y.append("Y")
    X.append([col for col in ds1.columns if col != "Y"])
    ds1.name = "DATASET_1"

    #http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
    header = [col for col in range(17)]
    ds2 = pd.read_csv("house-votes-84.data", names=header)
    Datasets.append(ds2)
    Y.append(0)
    X.append([col for col in ds2.columns if col != 0])
    ds2.name = "DATASET_2"

    #http://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    header = ["TLS", "TMS", "TRS", "MLS", "MMS", "MRS", "BLS", "BMS", "BRS", "Class"]
    ds3 = pd.read_csv("tic-tac-toe.data", names=header)
    Datasets.append(ds3)
    Y.append("Class")
    X.append([col for col in ds3.columns if col != "Class"])
    ds3.name = "DATASET_3"

    #http://archive.ics.uci.edu/ml/datasets/Breast+Cancer
    header = [col for col in range(10)]
    ds4 = pd.read_csv("breast-cancer.data", names=header)
    Datasets.append(ds4)
    Y.append(0)
    X.append([col for col in ds4.columns if col != 0])
    ds4.name = "DATASET_4"

    return Datasets, X, Y
