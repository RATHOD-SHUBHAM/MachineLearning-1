'''
Note :
    Use Python Version 3.6
    numpy
    pandas 0.21
    scipy


Using a wrong version leads to code crash


Note: Large data set takes a long time to run. almost 15-20 min


increase step value for more accuracy but thea time increases




Warning says: you try to take the natural logarithm of a negative number. This will result in a NaN.
Depending on the function used, you will be presented with a range of errors.
'''

# Todo: All the import Lib are here
import numpy as np
import pandas as pd
from typing import List
from scipy.spatial.distance import correlation


# todo: matrix factorization
'''
     Let's now use matrix factorization to do the same exercise ie
    # finding the recommendations for a user
    # The idea here is to identify some factors (these are factors which influence
    # a user'r rating). The factors are identified by decomposing the 
    # user item rating matrix into a user-factor matrix and a item-factor matrix
    # Each row in the user-factor matrix maps the user onto the hidden factors
    # Each row in the product factor matrix maps the item onto the hidden factors
    # This operation will be pretty expensive because it will effectively give us 
    # the factor vectors needed to find the rating of any product by any user 
    # (in the  previous case we only did the computations for 1 user)
    
    
    # R is the user item rating matrix 
    # K is the number of factors we will find 
    # We'll be using Stochastic Gradient descent to find the factor vectors 
    # steps, gamma and lamda are parameters the SGD will use - we'll get to them
    # in a bit 
'''
def matrixFactorization(R, K, steps=10, gamma=0.001, lamda=0.02):
    N = len(R.index)
    M = len(R.columns)
    P = pd.DataFrame(np.random.rand(N, K), index=R.index)
    Q = pd.DataFrame(np.random.rand(M, K), index=R.columns)
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    eij = R.loc[i, j] - np.dot(P.loc[i], Q.loc[j])
                    P.loc[i] = P.loc[i] + gamma * (eij * Q.loc[j] - lamda * P.loc[i])
                    Q.loc[j] = Q.loc[j] + gamma * (eij * P.loc[i] - lamda * Q.loc[j])
        e = 0
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    e = e + pow(R.loc[i, j] - np.dot(P.loc[i], Q.loc[j]), 2) + lamda * (
                            pow(np.linalg.norm(P.loc[i]), 2) + pow(np.linalg.norm(Q.loc[j]), 2))
        if e < 0.001:
            break
        print(step)
    return P, Q


# todo: computing Similarity between two users.
# WE DONT USE EUCLIDIAN DISTANCE INSTEAD WE USE CORRELATION HERE
def twoUser(usr1, usr2):
    usr1 = np.array(usr1) - np.nanmean(usr1)
    usr2 = np.array(usr2) - np.nanmean(usr2)
    movieIds = [i for i in range(len(usr1)) if usr1[i] > 0 and usr2[i] > 0]
    if len(movieIds) == 0:
        return 0
    else:
        usr1 = np.array([usr1[i] for i in movieIds])
        usr2 = np.array([usr2[i] for i in movieIds])
        return correlation(usr1, usr2)


# todo: Computing Nearest Neighbours
def nearestNeighbour(MatrixRating, activeUser: int, K: int):
    similarMatrix = pd.DataFrame(index=MatrixRating.index, columns=['Similarity'])
    for i in MatrixRating.index:
        similarMatrix.loc[i] = twoUser(MatrixRating.loc[activeUser], MatrixRating.loc[i])
    similarMatrix = pd.DataFrame.sort_values(similarMatrix, ['Similarity'], ascending=[0])

    nearestNeighbours = similarMatrix[:K]

    neighbourRatings = MatrixRating.loc[nearestNeighbours.index]

    predictItemRating = pd.DataFrame(index=MatrixRating.columns, columns=['Rating'])

    for i in MatrixRating.columns:
        # for each item
        predictedRating = np.nanmean(MatrixRating.loc[activeUser])
        # start with the average rating of the user
        for j in neighbourRatings.index:
            # for each neighbour in the neighbour list
            if MatrixRating.loc[j, i] > 0:
                predictedRating += (MatrixRating.loc[j, i] - np.nanmean(MatrixRating.loc[j])) * nearestNeighbours.loc[
                    j, 'Similarity']
        predictItemRating.loc[i, 'Rating'] = predictedRating
    return predictItemRating


# todo: Top Recommender [ RECOMMENDATION SYSTEM]
'''
# First we'll represent each user as a vector - each element of the vector 
# will be their rating for 1 movie. Since there are 1600 odd movies in all 
# Each user will be represented by a vector that has 1600 odd values 
# When the user doesn't have any rating for a movie - the corresponding 
# element will be blank. NaN is a value in numpy that represents numbers that don't 
# exist. This is a little tricky - any operation of any other number with NaN will 
# give us NaN. So, we'll keep this mind as we manipulate the vectors 

pandas pivot table is very much like an excel pivot table or an SQL group by
# This will take our table which is arranged like userid, itemid, rating 
# and give us a new table in which the row index is the userId, the column idex is
# the itemId, and the value is the rating 

'''


def topRec(matrixRating, movieData, activeUser: int, N: int, NumberOfNeighbours: int) -> List:
    predictRating = nearestNeighbour(matrixRating, activeUser, NumberOfNeighbours)
    moviesAlreadyWatched = list(matrixRating.loc[activeUser].loc[matrixRating.loc[activeUser] > 0].index)
    predictRating = predictRating.drop(moviesAlreadyWatched)
    topRecommendations = pd.DataFrame.sort_values(predictRating, ['Rating'], ascending=[0])[:N]
    topRecommendationTitles = (movieData.loc[movieData.movieId.isin(topRecommendations.index)])
    return list(topRecommendationTitles.title)


# todo: Top Favourite Movie
'''
    #1. subset the dataframe to have the rows corresponding to the active user
    # 2. sort by the rating in descending order
    # 3. pick the top N rows
'''


def topNfavoriteMovies(ratingData, activeUser: int, N: int) -> List:
    topMovies = pd.DataFrame.sort_values(ratingData[ratingData.userId == activeUser], ['rating'], ascending=[0])[:N]
    return list(topMovies.title)


# todo: ReadFile
def readFile(Filename):
    if Filename == 'smallratings.csv':
        ratingData = pd.read_csv(Filename, usecols=['userId', 'movieId', 'rating'])
        return ratingData
    else:
        movieData = pd.read_csv(Filename, usecols=['movieId', 'title'])
        return movieData


# todo: Main program where user will provide input
def main():
    # TODO: ENTER NUMBER OF NEAREST NEIGHBOUR SEARCH
    # enter =  10
    print("Example 10: So 10 neighbours will be selected")
    NumberOfNeighbours = int(input("Enter the number of neighbours you want to select: "))

    # TODO: READ FILE
    # read the rating csv file into a dataframe
    ratingFile = 'smallratings.csv'
    ratingData = readFile(ratingFile)
    # print(ratingData.head())

    # read movie csv file
    movieInfoFile = "smallmovies.csv"
    movieData = readFile(movieInfoFile)

    # print(movieData.head())

    # todo: joining column from different csv into single csv
    # adds title to rating data
    ratingData = pd.merge(ratingData, movieData, left_on='movieId', right_on="movieId")

    # fetch first 11 row + the column 'userID'
    # ratingData.loc[0:10, ['userId']]

    print('example: Toy Story (1995)', 'M*A*S*H (a.k.a. MASH) (1970)', 'Excalibur (1981)',
          'Indiana Jones and the Last Crusade (1989)')
    print("If given a wrong input -- > error will be popped")

    print("\n")

    MovieName = input("Enter A Movie Name Of Your Choice: ")
    toyStoryUsers = ratingData[ratingData.title == MovieName]
    # print(toyStoryUsers['userId'].values[0])

    # Sort data on columns [ 'userId', 'movieId' ]  in ascending order
    ratingData = pd.DataFrame.sort_values(ratingData, ['userId', 'movieId'], ascending=[0, 1])

    # todo: Similarity Matrix
    # panda pivot table function
    matrixRating = pd.pivot_table(ratingData, values='rating', index=['userId'], columns=['movieId'])
    print("Generating Matrix")
    print(matrixRating.head())

    activeUser = toyStoryUsers['userId'].values[0]

    print("\n\n")
    print("Example:  enter 3 , and top 3 fav movie will be selected")
    Nfav = int(input("Enter The Number Of fav movies you wanna select: "))

    print("\n")
    print("Example:  enter 5 , to get top 5 recommendation ")
    NumberOfRecommendationtobefetched = int(
        input("Enter the number of recommendation you wanna see based on your choice: "))
    print("\n")

    # TODO: PICK USERS TOP N FAV MOVIE
    print("User Fav Movie for which he has giving highest rating: ")
    print(topNfavoriteMovies(ratingData, activeUser, Nfav))

    print("\n")

    # TODO: RECOMMENDATION SYSTEM USING COLLABORATIVE APPROACH.
    # THIS WILL PRINT RECOMMENDATION BASED ON THE NEAREST NEIGHBOUR WITHOUT USING MATRIX DECOMPOSITION
    print("Top Recommended movie with KNN and without matrix factor are:  ")
    print(topRec(matrixRating, movieData, activeUser, NumberOfRecommendationtobefetched, NumberOfNeighbours))

    # TODO: RECOMMENDATION SYSTEM USING MATRIX DECOMPOSITION
    print("Predicting Movies using Matrix Factorizaton")
    '''
        # increase step value for more accuracy but the run time increases
        # matrix rating should run on entire dataset but that takes a lot of time so restricting it to first 100 users
        # steps loop through rating N times
    '''

    (P, Q) = matrixFactorization(matrixRating.iloc[:100, :100], K=2, gamma=0.001, lamda=0.02, steps=2)

    activeUser = toyStoryUsers['userId'].values[0]
    predictItemRating = pd.DataFrame(np.dot(P.loc[activeUser], Q.T), index=Q.index, columns=['Rating'])
    topRecommendations = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[
                         :NumberOfRecommendationtobefetched]

    # We found the ratings of all movies by the active user and then sorted them to find the top 3 movies
    topRecommendationTitles = movieData.loc[movieData.movieId.isin(topRecommendations.index)]
    print("\n\n")
    print("Movie Recommendation are: ")
    print(list(topRecommendationTitles.title))


if __name__ == '__main__':
    main()
