#! /usr/bin/env python

from sklearn import naive_bayes, tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sys
import numpy as np
from math import floor
from random import shuffle
from cross_creator import readAndSplit, printMatrix

def betterProba(probabilities):
    bigger = 0;
    rightLabel= 0;
    for label, prob in enumerate(probabilities):
        if bigger < prob:
            bigger = prob
            rightLabel = label
    return {'label': rightLabel + 1, 'prob': bigger, 'valid': True}

def serializeClassifiers(classifiers, thresholds, train, validation):
    finish = False
    i = 0
    completeList = [{'label': x, 'prob': 0.0, 'valid':False} for x in validation["labels"]]
    while not finish and i < len(thresholds):
        classifier = classifiers[i].fit(train["base"], train["labels"])
        matrix = classifier.predict_proba(validation["base"])
        tmp_values = [betterProba(x) for x in matrix]
        finish = True
        for idx in range (0, len(tmp_values)):
            if not completeList[idx]["valid"]:
                finish = False;
                if tmp_values[idx]['prob'] > thresholds[i]:
                    completeList[idx] = tmp_values[idx]
        i += 1

    return [x['label'] for x in completeList]

def sumMatrix (sum, m):
    for i in range (0, len(m)):
        for j in range (0, len(m[i])):
            sum[i][j] += m[i][j]

def normalizeMatrix (m, f):
    for i in range (0, len(m)):
        for j in range (0, len(m[i])):
            m[i][j] *= f

def accumulate(acc, key, precision, matrix):
    if key in acc:
        sumMatrix(acc[key]["matrix"], matrix)
        acc[key]["precision"] += precision
        acc[key]["wilcox"].append(str(precision))

    else:
        acc[key] = {"matrix": matrix, "precision": precision, "wilcox": [str(precision)]}

def printStats(stats, norm):
    for key, value in stats.iteritems():
        print(key)
        print("Matriz de confusao")
        normalizeMatrix(value["matrix"], (1.0/norm))
        printMatrix(value["matrix"])
        print("Precisao: %f" % (value["precision"] / norm))
        print("Precisao por repeticao (para Wilcox):\n")
        print(" ".join(value["wilcox"]))
        print("------------------------------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: %s <train base> <percentage> <repeats>' % sys.argv[0])
        sys.exit()

    statistics = {};

    for i in range (0, int(sys.argv[3])):
        cross = readAndSplit(sys.argv[1], float(sys.argv[2]))
        trainBase = cross["train"]
        validationBase = cross["validation"]
        print (trainBase) 
        print (validationBase) 


        #Naive Bayes
        gnb = naive_bayes.GaussianNB()
        gnb = gnb.fit(trainBase["base"], trainBase["labels"])
        values = gnb.predict(validationBase["base"])
        accumulate( statistics
                    , "Nayve Bayes"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Decision tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(trainBase["base"], trainBase["labels"])
        values = clf.predict(validationBase["base"])
        accumulate( statistics
                    , "Decision Tree"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Bagging of Decisions Trees
        bagging = BaggingClassifier(tree.DecisionTreeClassifier()
                                    , max_samples = 0.5, max_features = 0.5)
        bagging.fit(trainBase["base"], trainBase["labels"])
        values = bagging.predict(validationBase["base"])
        accumulate( statistics
                    , "Bagging (Decision Tree)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Bagging of Bayve Baies
        bagging = BaggingClassifier(naive_bayes.GaussianNB()
                                    , max_samples = 0.5, max_features = 0.5)
        bagging.fit(trainBase["base"], trainBase["labels"])
        values = bagging.predict(validationBase["base"])
        accumulate( statistics
                    , "Bagging (Nayve Bayes)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #SVM (RBF) Libsvm configuration
        svm = SVC(C = 32, kernel = 'rbf', gamma = 0.5)
        svm.fit(trainBase["base"], trainBase["labels"])
        values = svm.predict(validationBase["base"])
        accumulate( statistics
                    , "SVM (C = 32, RBF, Gamma = 0.5)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #KNN (default)
        knn = KNeighborsClassifier()
        knn.fit(trainBase["base"], trainBase["labels"])
        values = knn.predict(validationBase["base"])
        accumulate( statistics
                    , "KNN (default)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #KNN 3
        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(trainBase["base"], trainBase["labels"])
        values = knn.predict(validationBase["base"])
        accumulate( statistics
                    , "KNN (K = 3)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #KNN 7
        knn = KNeighborsClassifier(n_neighbors = 7)
        knn.fit(trainBase["base"], trainBase["labels"])
        values = knn.predict(validationBase["base"])
        accumulate( statistics
                    , "KNN (K = 7)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #RandomForests (default)
        knn = RandomForestClassifier()
        knn.fit(trainBase["base"], trainBase["labels"])
        values = knn.predict(validationBase["base"])
        accumulate( statistics
                    , "Random Forests (Default)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Bagging of KNN (default)
        bagging = BaggingClassifier(KNeighborsClassifier()
                                    , max_samples = 0.5, max_features = 0.5)
        bagging.fit(trainBase["base"], trainBase["labels"])
        values = bagging.predict(validationBase["base"])
        accumulate( statistics
                    , "Bagging (KNN)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))

        #Bagging of SVM
        bagging = BaggingClassifier(SVC(C = 32, kernel = 'rbf', gamma = 0.5)
                                    , max_samples = 0.5, max_features = 0.5)
        bagging.fit(trainBase["base"], trainBase["labels"])
        values = bagging.predict(validationBase["base"])
        accumulate( statistics
                    , "Bagging (SVM)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))

        #Voting Classifier Soft
        voting = VotingClassifier(estimators=[
                    ('knn', knn), ('decision', clf), ('gnb', gnb)],
                    voting='soft')
        voting.fit(trainBase["base"], trainBase["labels"])
        values = voting.predict(validationBase["base"])
        accumulate( statistics
                    , "Voting (Soft)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Voting Classifier Hard
        voting = VotingClassifier(estimators=[
                    ('knn', knn), ('gbb', gnb), ('decisiontree', clf)],
                    voting='hard')
        voting.fit(trainBase["base"], trainBase["labels"])
        values = voting.predict(validationBase["base"])
        accumulate( statistics
                    , "Voting (Hard)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Decision tree (Log)
        clf = tree.DecisionTreeClassifier(max_features = "log2")
        clf = clf.fit(trainBase["base"], trainBase["labels"])
        values = clf.predict(validationBase["base"])
        accumulate( statistics
                    , "Decision Tree Log"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Decision tree (Sqrt)
        clf = tree.DecisionTreeClassifier(max_features = "sqrt")
        clf = clf.fit(trainBase["base"], trainBase["labels"])
        values = clf.predict(validationBase["base"])
        accumulate( statistics
                    , "Decision Tree (Sqrt)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Serialize (SVM, NayveBayes, KNN)
        values = serializeClassifiers([
            SVC(C = 32, kernel = 'rbf', gamma = 0.5, probability = True),
            naive_bayes.GaussianNB(),
            KNeighborsClassifier()
        ], [0.9, 0.6, 0.0], trainBase, validationBase)
        accumulate( statistics
                    , "Serialize (SVM, NayveBayes, KNN)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
        #Serialize (SVM, Decision Treem KNN)
        values = serializeClassifiers([
            SVC(C = 32, kernel = 'rbf', gamma = 0.5, probability = True),
            tree.DecisionTreeClassifier(),
            KNeighborsClassifier()
        ], [0.9, 0.6, 0.0], trainBase, validationBase)
        accumulate( statistics
                    , "Serialize (SVM, Decision Tree, KNN)"
                    , accuracy_score(validationBase["labels"], values)
                    , confusion_matrix(validationBase["labels"], values))
    printStats(statistics, int(sys.argv[3]))
    sys.exit()
