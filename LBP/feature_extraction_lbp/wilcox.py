#! /usr/bin/env python
import sys
from scipy.stats import  wilcoxon
import numpy as np

def genTable(filename, n):
    i = 0;
    labels = [""]
    clfs = []
    with open(filename, "r") as f:
        for line in f:
            if i == 0:
                labels.append(line.split("\n")[0])
            elif i == (n+5):
                clfs.append(np.array(line.split(' ')).astype(np.float))
            i = (i +1) % (n+8)


    txt = [ p for p in labels]
    print(";".join(txt))
    for i in range(0, len(clfs)):
        res = [labels[i+1]];
        for j in range(0, len(clfs)):
            statistics, pvalue = wilcoxon(clfs[i], clfs[j])
            res.append("%.5f" % pvalue);
        txt = [p for p in res]
        print((";".join(res)))
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: %s <file_in> <labels>' % sys.argv[0])
        sys.exit()
    genTable(sys.argv[1], int(sys.argv[2]))
