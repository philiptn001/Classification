# Classification
Project to learn about Prediction techniques using classification algorithms

## Project 1:
Project to train and test a KNN classifier
Steps:
* Load the iris dataset
* Split the dataset into test and train datasets; 70% of the data should be used for training the classifier and the rest for testing
* Train a KNN classifier by fitting on the train dataset
* Based on the trained classifier, predict the classes of the test dataset;
* Print the predictions and the real class labels

## Project 2:
Project to evaluate KNN classifier over iris dataset
Steps:
* Calculate and print the accuracy of the classifier
* Print the confusion_matrix
* Calculate and print the mean precision , and recall

## Project 3:
Project to learn to evaluate various classifiers using k-fold cross evaluation
Steps:
* Load the iris dataset
* Perform a 5-fold cross validation over the following classifiers, and record their accuracy:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

KNeighborsClassifier()
DecisionTreeClassifier()
LinearDiscriminantAnalysis()
LogisticRegression()
GaussianNB()
SVC()
    
* Sort the classifiers based on their accuracy
* Print the classifiers and their accuracy in order.