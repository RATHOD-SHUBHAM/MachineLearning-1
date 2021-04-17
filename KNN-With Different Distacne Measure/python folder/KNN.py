# Machine Learning Programming Assignment
# Name: Shubham Shankar
# UTA ID: 1001761068
# Sec: 2208-CSE-6363-004


# Todo : KNN Algorithm
import pandas as pd
import random as r
from random import seed
from math import sqrt


# Todo: Function to read .csv file
def read_csv(file_name):
    df = pd.read_csv(file_name, delimiter=',', header=None)
    #     print(df)
    dataset = [list(row) for row in df.values]
    dataset.insert(0, df.columns.to_list())
    return dataset


# Todo : Function to convert column values into float
def str_column_into_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


#     print('\n\n')
#     print(type(row[column]))


# Todo : Functions to convert column values to int
def str_column_into_int(dataset, column):
    for row in dataset:
        row[column] = int(row[column])


# todo : Function to convert class value into numeric integer
def stringColumn_to_integer(dataset, column):
    # column = 4 ie class
    class_values = [row[column] for row in dataset]
    #     print("the class_values is: ",class_values)
    #     print(type(class_values))
    #     print('\n\n')
    unique_class_values = set(class_values)
    #     print(unique_class_values)
    look_up_for = dict()
    for i, values in enumerate(unique_class_values):
        #         print('\n\n')
        #         print(i,values)
        look_up_for[values] = i
        print('[%s] => %d' % (values, i))
    #         print(look_up_for)
    for row in dataset:
        row[column] = look_up_for[row[column]]
    return look_up_for


# Todo: K Fold Cross Validation
def k_fold_cross_validation(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    #     print(len(dataset)) == 150
    fold_size = int(len(dataset) / n_folds)
    #     print(fold_size) == 30
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = r.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        #             https://www.geeksforgeeks.org/python-list-pop
        #             print("fold value is: ",fold)
        dataset_split.append(fold)
    return dataset_split


# Todo: Euclidian Distance
def euclidean_distance(a, b):
    a = a[:-1]
    b = b[:-1]
    return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(a, b)))


# Todo: Hamming Distance
def hamming_distance(a, b):
    a = a[:-1]
    b = b[:-1]
    return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)


# Todo : Manhattan Distance
def manhattan_distance(a, b):
    a = a[:-1]
    b = b[:-1]
    return sum(abs(e1 - e2) for e1, e2 in zip(a, b))


# Todo : Minkowski Distance
def minkowski_distance(a, b, p):
    a = a[:-1]
    b = b[:-1]
    return sum(abs(e1 - e2) ** p for e1, e2 in zip(a, b)) ** (1 / p)


# Todo: Function to get Neighbour
def get_neighbors(train_set, test_row, number_of_neighbours):
    euclidian_distances = list()
    hamming_distances = list()
    manhattan_distances = list()
    minkowski_distances = list()
    p = 2

    for train_row in train_set:
        euclidean_dist = euclidean_distance(test_row, train_row)
        hamming_dist = hamming_distance(test_row, train_row)
        manhattan_dist = manhattan_distance(test_row, train_row)
        minkowski_dist = minkowski_distance(test_row, train_row, p)

        euclidian_distances.append((train_row, euclidean_dist))
        hamming_distances.append((train_row, hamming_dist))
        manhattan_distances.append((train_row, manhattan_dist))
        minkowski_distances.append((train_row, minkowski_dist))

    euclidian_distances.sort(key=lambda tup: tup[1])
    hamming_distances.sort(key=lambda tup: tup[1])
    manhattan_distances.sort(key=lambda tup: tup[1])
    minkowski_distances.sort(key=lambda tup: tup[1])

    euclidian_neighbors = list()
    hamming_neighbors = list()
    manhattan_neighbors = list()
    minkowski_neighbors = list()
    for i in range(number_of_neighbours):
        euclidian_neighbors.append(euclidian_distances[i][0])
        hamming_neighbors.append(hamming_distances[i][0])
        manhattan_neighbors.append(manhattan_distances[i][0])
        minkowski_neighbors.append(minkowski_distances[i][0])

    return euclidian_neighbors, hamming_neighbors, manhattan_neighbors, minkowski_neighbors


# Todo: Function to predict classifier
def predict_classification(train_set, row, number_of_neighbours):
    euclid_neighbors, hamming_neighbors, manhattan_neighbors, minkowski_neighbors = get_neighbors(train_set, row,
                                                                                                  number_of_neighbours)

    euclid_output = [row[-1] for row in euclid_neighbors]
    euclid_prediction = max(set(euclid_output), key=euclid_output.count)

    hamming_output = [row[-1] for row in hamming_neighbors]
    hamming_prediction = max(set(hamming_output), key=hamming_output.count)

    manhattan_output = [row[-1] for row in manhattan_neighbors]
    manhattan_prediction = max(set(manhattan_output), key=manhattan_output.count)

    minkowski_output = [row[-1] for row in minkowski_neighbors]
    minkowski_prediction = max(set(minkowski_output), key=minkowski_output.count)

    return euclid_prediction, hamming_prediction, manhattan_prediction, minkowski_prediction


# Todo: KNN Algorithm
def k_nearest_neighbors(train_set, test_set, number_of_neighbours):
    euclid_predictions = list()
    hamming_predictions = list()
    manhattan_predictions = list()
    minkowski_predictions = list()

    for row in test_set:
        euclid_predict_output, hamming_predict_output, manhattan_predict_output, minkowski_predict_output = predict_classification(
            train_set, row, number_of_neighbours)

        euclid_predictions.append(euclid_predict_output)
        hamming_predictions.append(hamming_predict_output)
        manhattan_predictions.append(manhattan_predict_output)
        minkowski_predictions.append(minkowski_predict_output)

    return euclid_predictions, hamming_predictions, manhattan_predictions, minkowski_predictions


# Todo: Function to calculate Accuracy Metrix
def accuracy_metrix(actual, prediction):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Todo: Evaluation Function
def eval_algorithm(dataset, k_nearest_neighbors, n_folds, *args):
    k_fold_data_set = k_fold_cross_validation(dataset, n_folds)

    euclid_prediction_score = list()
    hamming_prediction_score = list()
    manhattan_prediction_score = list()
    minkowski_prediction_score = list()

    for fold in k_fold_data_set:
        train_set = list(k_fold_data_set)
        train_set.remove(fold)  # https://www.programiz.com/python-programming/methods/list/remove
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        euclid_prediction, hamming_prediction, manhattan_prediction, minkowski_prediction = k_nearest_neighbors(
            train_set, test_set, *args)

        actual = [row[-1] for row in fold]

        euclid_accuracy = accuracy_metrix(actual, euclid_prediction)
        euclid_prediction_score.append(euclid_accuracy)

        hamming_accuracy = accuracy_metrix(actual, hamming_prediction)
        hamming_prediction_score.append(hamming_accuracy)

        manhattan_accuracy = accuracy_metrix(actual, manhattan_prediction)
        manhattan_prediction_score.append(manhattan_accuracy)

        minkowski_accuracy = accuracy_metrix(actual, minkowski_prediction)
        minkowski_prediction_score.append(minkowski_accuracy)

    return euclid_prediction_score, hamming_prediction_score, manhattan_prediction_score, minkowski_prediction_score


# Todo: The Main Function
def main():
    seed(1)  # generate the same random value each time i run my code rather then a new number
    print("Enter:  1 ---> car dataset, 2 -----> hayes-roth dataset, 3 -----> breast-cancer dataset, 4 ---> irish dataset")
    print("\n")
    input_file = input("enter a number for dataset: ")

    if (input_file == '1'):
        print("The Dataset is car")
        file_to_be_read = 'car.csv'
        dataset = read_csv(file_to_be_read)
        dataset = dataset[1:]
        print("\n\n")
        #         print(dataset)
        print(" Vectorized form ")
        print(" ")
        for column in range(len(dataset[0]) - 1):
            #             print("the column is : ",column)
            stringColumn_to_integer(dataset, column)
        print("\n\n")
        print("class values")
        print(" ")
        stringColumn_to_integer(dataset, len(dataset[0]) - 1)


    elif (input_file == '2'):
        print("The Dataset is hayes-roth")
        file_to_be_read = 'hayes-roth.csv'
        dataset = read_csv(file_to_be_read)
        dataset = dataset[1:]
        print("\n\n")
        #         print(dataset)
        print(" Vectorized form ")
        print(" ")
        for column in range(len(dataset[0]) - 1):
            str_column_into_int(dataset, column)
        print("\n\n")
        print("class values")
        print(" ")
        stringColumn_to_integer(dataset, len(dataset[0]) - 1)

    elif (input_file == '3'):
        print("The Dataset is breast-cancer")
        file_to_be_read = 'breast-cancer.csv'
        dataset = read_csv(file_to_be_read)
        dataset = dataset[1:]
        print("\n\n")
        #         print(dataset)
        print(" Vectorized form ")
        print(" ")
        for column in range(len(dataset[0]) - 1):
            stringColumn_to_integer(dataset, column)
        print("\n\n")
        print("class values")
        print(" ")
        stringColumn_to_integer(dataset, len(dataset[0]) - 1)

    elif (input_file == '4'):
        print("The Dataset is irish")
        file_to_be_read = 'irish.csv'
        dataset = read_csv(file_to_be_read)
        dataset = dataset[1:]
        print("\n\n")
        #         print(dataset)
        print(" Vectorized form ")
        print(" ")
        for column in range(len(dataset[0]) - 1):
            # convert the column with string values into float [col:0,1,2,3]
            str_column_into_float(dataset, column)
        print("class values")
        print(" ")
        stringColumn_to_integer(dataset, len(dataset[0]) - 1)

    #     print(dataset)
    # algorithm
    n_folds = 10
    number_of_neighbours = 5

    euclid_prediction_score, hamming_prediction_score, manhattan_prediction_score, minkowski_prediction_score = eval_algorithm(
        dataset, k_nearest_neighbors, n_folds, number_of_neighbours)
    print("\n\n")
    print("Euclid Predicted Score is: %s" % euclid_prediction_score)
    print("")
    print("Euclid Mean of accuracy: %.3f%%" % (sum(euclid_prediction_score) / float(len(euclid_prediction_score))))
    print("\n\n")

    print("\n\n")
    print("Hamming Predicted Score is: %s" % hamming_prediction_score)
    print("")
    print("Hamming Mean of accuracy: %.3f%%" % (sum(hamming_prediction_score) / float(len(hamming_prediction_score))))
    print("\n\n")

    print("\n\n")
    print("Manhattan Predicted Score is: %s" % manhattan_prediction_score)
    print("")
    print("Manhattan Mean of accuracy: %.3f%%" % (
            sum(manhattan_prediction_score) / float(len(manhattan_prediction_score))))
    print("\n\n")

    print("\n\n")
    print("Minkowski Predicted Score is: %s" % minkowski_prediction_score)
    print("")
    print("Minkowski Mean of accuracy: %.3f%%" % (
            sum(minkowski_prediction_score) / float(len(minkowski_prediction_score))))
    print("\n\n")


#     if(input_file == '1'):
#         # new row
#         row = [0, 0, 0, 0, 1, 2]
#         # ['vhigh', 'vhigh', '2', '2', 'small', 'low']
# #         the predicted class should be : [unacc]
#         # predict the label
#         label = predict_classification(dataset,row,number_of_neighbours)
#         print('Data=%s, Predicted class is : %s' % (row, label))
#     elif(input_file == '2'):
#         # new row
#         row = [92, 2, 1, 1, 2]
#         # prediction class should be 1
#         # predict the label
#         label = predict_classification(dataset,row,number_of_neighbours)
#         print('Data=%s, Predicted class is : %s' % (row, label))
#     elif(input_file == '3'):
#         # new row
#         row = [0, 5, 0, 10, 2, 0, 2, 1, 4]
# #             no
#         # predict the label
#         label = predict_classification(dataset,row,number_of_neighbours)
#         print('Data=%s, Predicted class is : %s' % (row, label))
#     elif(input_file == '4'):
#         # new row
#         row = [5.1, 3.5, 1.4, 0.2]
# #          'Iris-setosa'
#         # predict the label
#         label = predict_classification(dataset,row,number_of_neighbours)
#         print('Data=%s, Predicted class is : %s' % (row, label))

if __name__ == '__main__':
    main()
