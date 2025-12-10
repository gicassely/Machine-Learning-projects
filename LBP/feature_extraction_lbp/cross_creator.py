#! /usr/bin/env python

from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import numpy as np
from math import floor
from random import shuffle

def readAndSplit (filename, percent):
    """
        Open a file with the format
        nlines dims
        float float ...(dims times) int(label)
        .
        .
        .
        (nlines times)

        returns a dictionay with.
        base: a list of lists of size nlines where each element
        has size dims parsed to float
        labels: a list nlines size with the labels parsed to int
        nlines : number of lines
        dims : dimentions
    """
    f = open(filename, "r")
    args = f.readline().split(' ')
    nlines = int(args[0])
    dims = int(args[1])
    classes = {};
    train = {}
    validation = {}
    r = {
        "train": { "base": [], "labels": [], "nlines": 0, "dims": dims}
        , "validation": { "base": [], "labels": [], "nlines": 0, "dims": dims}
    }
    for i in range (0, nlines):
        line = f.readline().split(' ')
        label = line.pop()
        floatLine = np.array(line).astype(np.float)
        if label in classes:
            classes[label].append(floatLine)
        else:
            classes[label] = [floatLine]
            train[label] = []
            validation[label] = []
    emptyClasses = 0
    classesLength = len(classes)
    for key, value in classes.items():
        shuffle(value);
    for key, value in classes.items():
        border = int(floor(len(value) * percent))
        train[key] = value[:border]
        validation[key] = value[border:]
    while emptyClasses < classesLength:
        for key, value in train.items():
            if len(value) == 0:
                del train[key]
                emptyClasses += 1
            else:
                r["train"]["base"].append(value.pop())
                r["train"]["labels"].append(int(key))
                r["train"]["nlines"] += 1

    emptyClasses = 0
    while emptyClasses < classesLength:
        for key, value in validation.items():
            if len(value) == 0:
                del validation[key]
                emptyClasses += 1
            else:
                r["validation"]["base"].append(value.pop())
                r["validation"]["labels"].append(int(key))
                r["validation"]["nlines"] += 1

    r["labels"] = classesLength
    return r

def printMatrix (m):
    for i in m:
        print(" ".join(str(x) for x in i))

def printBase (b, filename):
    file = open(filename, "w")
    print ("%d %d" % (b["nlines"], b["dims"]))
    file.write ("%d %d\n" % (b["nlines"], b["dims"]))
    for i in range (0, b["nlines"]):
        line = " ".join(str(x) for x in b["base"][i])
        print ("%s %d" % (line, b["labels"][i]))
        file.write ("%s %d\n" % (line, b["labels"][i]))
    print("\n---------------------------------\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: %s <train base> <folds>' % sys.argv[0])
        sys.exit()
    cross = readAndSplit(sys.argv[1], float(sys.argv[2]))
    printBase (cross["train"], "train.out")
    printBase (cross["validation"], "validation.out")
    sys.exit()
