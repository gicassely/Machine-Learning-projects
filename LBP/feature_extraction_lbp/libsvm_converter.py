#! /usr/bin/env python
import sys

def convertBase (filename, outFile):
    f = open(filename, "r")
    out = open(outFile, "w")
    args = f.readline().split(' ')
    nlines = int(args[0])
    dims = int(args[1])
    for i in range (0, nlines):
        line = f.readline().split(' ')
        label = int(line.pop())
        convertedLine = []
        for key, value in enumerate(line):
            convertedLine.append("%d:%s" % ((key +1), value))
        lineAsString = "%d %s\n" % (label, " ".join(convertedLine))
        out.write(lineAsString)
    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: %s <base_in> <base_out>' % sys.argv[0])
        sys.exit()
    convertBase(sys.argv[1], sys.argv[2])
    sys.exit()
