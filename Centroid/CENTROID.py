import pandas as pd
import sklearn as sk
import re
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
import matplotlib
import matplotlib.pyplot as plt
# % matplotlib
# inline


# CLASSIFIERS
def LogReg(X, y, X_test, y_test):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500).fit(X, y)
    y_pred = clf.predict(X_test)
    return y_pred, accuracy_score(y_test, y_pred)


def SVClassifier(X, y, X_test, y_test):
    clf = SVC(gamma='auto')
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    return y_pred, accuracy_score(y_test, y_pred)


def knn(X, y, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    y_pred = neigh.predict(X_test)
    return y_pred, accuracy_score(y_test, y_pred)


# DATA HANDLERS
def saveCSV(trainX, trainY):
    trainX.to_csv('trainX.csv')
    trainY.to_csv('trainY.csv')


def pickDataClass(filename, class_ids):
    dataset = pd.read_csv(filename, header=None)
    dataset_t = dataset.T
    return dataset_t.loc[dataset_t[0].isin(class_ids)]


def letter_2_digit_convert(ids):
    return [ord(c) - 97 + 1 for c in ids.lower()]


def splitData2TestTrain(dataset, number_per_class, test_instances):
    classes = list(dataset.iloc[:, 0].unique())
    columns = dataset.columns.stop
    train = pd.DataFrame([], columns=[i for i in range(0, columns)])
    test = pd.DataFrame([], columns=[i for i in range(0, columns)])

    for cls in classes:
        class_data = dataset[dataset[0] == cls]
        #         print(test_instances)
        #         print(class_data[test_instances[0]:test_instances[1]])
        test = test.append(class_data[test_instances[0]:test_instances[1]])
        train = train.append(class_data[0:test_instances[0]])
        train = train.append(class_data[test_instances[1]:number_per_class])
    #         print(test)

    # Save train and test
    saveCSV(train.T, test.T)

    #     split X and yt
    test_attr = test.iloc[:, 1:columns]
    test_class = test.iloc[:, 0]
    test_class = test_class.astype('int')
    train_attr = train.iloc[:, 1:columns]
    train_class = train.iloc[:, 0]
    train_class = train_class.astype('int')
    return train_attr, train_class, test_attr, test_class


# ADDITIONAL HANDLERS
def shuffleSplitData2TestTrain(dataset, ratio=0.2):
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    rows = len(dataset)
    train = dataset.iloc[0:int(rows * (1 - ratio))]
    test = dataset.iloc[int(rows * (1 - ratio)):]
    train_attr, train_class, test_attr, test_class = train.iloc[:, 1:], train.iloc[:, 0], test.iloc[:, 1:], test.iloc[:,
                                                                                                            0]
    return train_attr, train_class, test_attr, test_class


# Shuffle Handwritten Dataset

# filename = "HandWrittenLetters.txt"
# # dataset = pickDataClass(filename, classes)
# dataset = pd.read_csv(filename, header=None)
# # ratio is for test percent
# ratio = 0.2
# train_attr, train_class, test_attr, test_class = shuffleSplitData2TestTrain(dataset, ratio)


# log = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500)
# log.fit(train_attr, train_class)
# ypred = log.predict(test_attr)
# accuracy = accuracy_score(list(test_class), ypred)
# print(ypred)
# print(accuracy)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


class Centroid:
    def fit(self, X, y):
        self.meanX = pd.DataFrame([], columns=X.columns)
        classes = list(y.unique())
        self.meanY = pd.Series(classes)
        for cls in classes:
            self.meanX = self.meanX.append(X[y == cls].mean(), ignore_index=True)

    def predict(self, X_test):
        pred_list = []
        for ri, row in X_test.iterrows():
            min_dist = float('inf')
            min_ind = -1
            for mean_index, mean_val in self.meanX.iterrows():
                dist = self.euclidean_distance(mean_val, row)
                if (dist < min_dist):
                    min_dist = dist
                    min_ind = mean_index
            pred_list.append(self.meanY[min_ind])
        return pred_list

    # Calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(1, len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)


# QUESTION A
print("QUESTION A")
result = {}
filename = "HandWrittenLetters.txt"
print("INPUT CLASSES FROM A TO Z . EG ABXCS")
classes = input("ENTER THE CLASS INPUT: ")
classes = letter_2_digit_convert(classes)
dataset = pickDataClass(filename, classes)
number_per_class = 39
k = 2
test_instances = "30:39"
test_instances = [int(i) for i in test_instances.split(':')]
result = {}
train_attr, train_class, test_attr, test_class = splitData2TestTrain(dataset, number_per_class, test_instances)

ypred, accuracy = SVClassifier(train_attr, train_class, test_attr, test_class)
result["svm"] = {
    "ypred": ypred,
    "accuracy": accuracy
}

ypred, accuracy = LogReg(train_attr, train_class, test_attr, test_class)
result["lin"] = {
    "ypred": ypred,
    "accuracy": accuracy
}

ypred, accuracy = knn(train_attr, train_class, test_attr, test_class)
result["knn"] = {
    "ypred": ypred,
    "accuracy": accuracy
}

clf = Centroid()
clf.fit(train_attr, train_class)
ypred = clf.predict(test_attr)
result["centroid"] = {
    "ypred": ypred,
    "accuracy": accuracy_score(ypred, test_class)
}

print("Question A RESULT: ")
print("SVM")
print(result["svm"])
print("Logistic Regression")
print(result["lin"])
print("KNN Regression")
print(result["knn"])
print("Centroid Regression")
print(result["centroid"])

# QUESTION B


result = {}
print("QUESTION B RUNNING ... ")
filename = "ATNTFaceImages400.txt"
dataset = pd.read_csv(filename, header=None)
dataset = dataset.T
folds = StratifiedKFold(n_splits=5)

log = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500)
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(gamma='auto')
cent = Centroid()

score_knn = []
score_svm = []
score_log = []
score_cent = []

X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
for train_index, test_index in folds.split(X, y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    score_knn.append(get_score(log, X_train, X_test, y_train, y_test))
    score_svm.append(get_score(svm, X_train, X_test, y_train, y_test))
    score_log.append(get_score(knn, X_train, X_test, y_train, y_test))
    cent.fit(X_train, y_train)
    score_cent.append(accuracy_score(cent.predict(X_test), y_test))

print("KNN 5 fold accuracies")
print(score_knn)
print("MEAN: ", sum(score_knn) / len(score_knn))
print("SVM 5 fold accuracies")
print(score_svm)
print("MEAN: ", sum(score_svm) / len(score_svm))
print("LOGISTIC REGRESSION 5 fold accuracies")
print(score_log)
print("MEAN: ", sum(score_log) / len(score_log))
print("Centroid Method 5 fold accuracies")
print(score_cent)
print("MEAN: ", sum(score_cent) / len(score_cent))

# QUESTION C

print("QUESTION C EXECUTING....")
result = {}
filename = "HandWrittenLetters.txt"
classes = "ABCDEFGHIJ"
classes = letter_2_digit_convert(classes)
dataset = pickDataClass(filename, classes)
number_per_class = 39
test_instances = [[0, 5], [0, 10], [0, 15], [0, 20], [0, 25], [0, 30], [0, 35]]
accuracies = []
print("Testing for 7 different ranges of train and test data")
for it, test in enumerate(test_instances):
    print('Iteration ', it)
    train_attr, train_class, test_attr, test_class = splitData2TestTrain(dataset, number_per_class, test)
    centroid = Centroid()
    centroid.fit(train_attr, train_class)
    y_pred = centroid.predict(test_attr)
    accuracies.append(accuracy_score(test_class, y_pred))

train_leng = [number_per_class - i[1] for i in test_instances]
plt.plot(train_leng, accuracies)
plt.ylabel('Accuracy')
plt.xlabel('No of training data')
plt.show()

# QUESTION D

print("entered into question D")
result = {}
filename = "HandWrittenLetters.txt"
classes = "KLMNOPQRST"
classes = letter_2_digit_convert(classes)
dataset = pickDataClass(filename, classes)
number_per_class = 39
test_instances = [[0, 5], [0, 10], [0, 15], [0, 20], [0, 25], [0, 30], [0, 35]]
accuracies = []
for test in test_instances:
    train_attr, train_class, test_attr, test_class = splitData2TestTrain(dataset, number_per_class, test)
    centroid = Centroid()
    centroid.fit(train_attr, train_class)
    y_pred = centroid.predict(test_attr)
    accuracies.append(accuracy_score(test_class, y_pred))

train_leng = [number_per_class - i[1] for i in test_instances]
plt.plot(train_leng, accuracies)
plt.ylabel('Accuracy')
plt.xlabel('No of training data')
plt.show()