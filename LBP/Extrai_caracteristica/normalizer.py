#! /usr/bin/env python
import sys
import numpy as np

def convertBase (filename, outFile):
    f = open(filename, "r")
    out = open(outFile, "w")
    args = f.readline().split(' ')
    nlines = int(args[0])
    dims = int(args[1])
    line = f.readline().split(' ');
    maxLabel = int(line.pop())
    maxLine = np.array(line).astype(np.float)
    minLine = np.copy(maxLine)
    base = [np.copy(maxLine)];
    labels = [maxLabel]
    out.write("%d %d\n" % (nlines, dims))
    for i in range (1, nlines):
        line = f.readline().split(' ')
        label = int(line.pop())
        floatLine = np.array(line).astype(np.float)
        for j in range (0, dims):
            if maxLine[j] < floatLine[j]:
                maxLine[j] = floatLine[j]
            if minLine[j] > floatLine[j]:
                minLine[j] = floatLine[j]
        base.append(floatLine);
        labels.append(label);
    for i in range(0, nlines):
        convertedLine = [
            str((base[i][j] - minLine[j])/(maxLine[j] - minLine[j]))
             for j in range(0, dims)
        ]
        lineAsString = "%s %d\n" % (" ".join(convertedLine), labels[i])
        out.write(lineAsString)
    convertedLine = [ str(item) for item in minLine]
    print("%s\n" % " ".join(convertedLine))
    convertedLine = [ str(item) for item in maxLine]
    print("%s\n" % " ".join(convertedLine))
    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: %s <base_in> <base_out>' % sys.argv[0])
        sys.exit()
    convertBase(sys.argv[1], sys.argv[2])
    sys.exit()
