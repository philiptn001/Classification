import pandas as pd
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score

if __name__ == '__main__':

    df = pd.read_csv('iris.csv')

    df = shuffle(df)

    x_df = df.drop('species', axis=1).values
    y_df = df['species'].values

    split_percentage=0.7

    split_df = int(len(x_df)*split_percentage)
    x_df_train = x_df[:split_df]
    y_df_train = y_df[:split_df]
    x_df_test = x_df[split_df:]
    y_df_test = y_df[split_df:]

    warnings.filterwarnings("ignore", category=FutureWarning)

    classifier = [KNeighborsClassifier(),
                DecisionTreeClassifier(),
                LinearDiscriminantAnalysis(),
                LogisticRegression(),
                GaussianNB(),
                SVC()]

    accuracy_list = []

    for i, classifier in enumerate(classifier):
        #dataset split to 5 fold
        acuracy = cross_val_score(classifier, x_df, y_df, cv=5)
        accuracy_list.append((acuracy.mean(), type(classifier).__name__))

    accuracy_list = sorted(accuracy_list, reverse=True)
    for i in accuracy_list:
        print(i[1], ':', i[0])

    # train using KNeighborsClassifier
    clas = KNeighborsClassifier()
    clas.fit(x_df_train, y_df_train)

    # predict the test set
    predictions = clas.predict(x_df_test)
    print("\n\n ___KNeighborsClassifier___\n\n")
    print("confusion_matrix:\n", confusion_matrix(y_df_test, predictions))
    print("precision:\t", precision_score(y_df_test, predictions, average=None))
    print("recall:\t\t", recall_score(y_df_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(y_df_test, predictions))

    
    # train using DecisionTreeClassifier
    clas = DecisionTreeClassifier()
    clas.fit(x_df_train, y_df_train)

    # predict the test set
    predictions = clas.predict(x_df_test)
    print("\n\n ___DecisionTreeClassifier___\n\n")
    print("confusion_matrix:\n", confusion_matrix(y_df_test, predictions))
    print("precision:\t", precision_score(y_df_test, predictions, average=None))
    print("recall:\t\t", recall_score(y_df_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(y_df_test, predictions))

    # train using LinearDiscriminantAnalysis
    clas = LinearDiscriminantAnalysis()
    clas.fit(x_df_train, y_df_train)

    # predict the test set
    predictions = clas.predict(x_df_test)
    print("\n\n ___LinearDiscriminantAnalysis___\n\n")
    print("confusion_matrix:\n", confusion_matrix(y_df_test, predictions))
    print("precision:\t", precision_score(y_df_test, predictions, average=None))
    print("recall:\t\t", recall_score(y_df_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(y_df_test, predictions))

    # train using LogisticRegression
    clas = LogisticRegression()
    clas.fit(x_df_train, y_df_train)

    # predict the test set
    predictions = clas.predict(x_df_test)
    print("\n\n ___LogisticRegression___\n\n")
    print("confusion_matrix:\n", confusion_matrix(y_df_test, predictions))
    print("precision:\t", precision_score(y_df_test, predictions, average=None))
    print("recall:\t\t", recall_score(y_df_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(y_df_test, predictions))

    # train using GaussianNB
    clas = GaussianNB()
    clas.fit(x_df_train, y_df_train)

    # predict the test set
    predictions = clas.predict(x_df_test)
    print("\n\n ___GaussianNB___\n\n")
    print("confusion_matrix:\n", confusion_matrix(y_df_test, predictions))
    print("precision:\t", precision_score(y_df_test, predictions, average=None))
    print("recall:\t\t", recall_score(y_df_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(y_df_test, predictions))

    # train using SVC
    clas = SVC()
    clas.fit(x_df_train, y_df_train)

    # predict the test set
    predictions = clas.predict(x_df_test)
    print("\n\n ___SVC___\n\n")
    print("confusion_matrix:\n", confusion_matrix(y_df_test, predictions))
    print("precision:\t", precision_score(y_df_test, predictions, average=None))
    print("recall:\t\t", recall_score(y_df_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(y_df_test, predictions))