# Centroid

This Project has 1 task. See "Task" below

In ATNT50 directory we have

trainDataXY.txt   

It contains 45 images. image 1-9 from class 1. image 10-18 from class 2. etc.
Each image is a column. 1st row are class labels.

testDataXY.txt     

It contain 5 images. 
Each image is a column. 1st row are class labels.

------------------------------------------------------------------------------------
You train the classifier using training data. Once classifier is trained,
you classifier the data in testData, and compare the obtained class labels 
to the ground-truth label provided there. 

These two data are simple training and testing data.
They are warm-up data, so you can see how your classifier work on this simple data. 

-------------------------------------------------------------------------------------


data set: ATNT-face-image400.txt  :

Text file. 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=28x23).

Total 40 classes. each class has 10 images. Total 40*10=400 images

----------------------------------------------------------------------------------------

data set: Hand-written-26-letters.txt :

Text file. 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=20x16).

Total 26 classes. each class has 39 images. Total 26*39=1014 images.


-------------------------------------------------------------------------------------
Once you are confident that your classifier works correctly,
you are to use 5-fold cross-validation (CV) assess/evaluate the classifier.
You do CV using the following two full datasets.

ATNT face images data are generally easier, i.e., you get high classification accuracy.

You run classifier on ATNT data first, to make sure you get correct results.

Hand-written-letters data are harder to classify, i.e., you get lower classification accuracy.

You run classifier on hand-written-letter data only after you are confident 
that your classifier works correctly.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Your task  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Purpose:
(1)  Implement Centroid methods by yourself.
(A£©
Use the data-handler to select classes from the hand-written-letter data. 
From this smaller dataset, Generate a training and test data: for each class
using the first 30 images for training and the remaining 9 images for test.
Do classification on the generated data using the classifers.

