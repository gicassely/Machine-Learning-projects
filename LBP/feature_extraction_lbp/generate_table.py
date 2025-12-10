#! /usr/bin/env python
import sys

def genTable(filename, fileout):
    f = open(filename, "r")
    out = open(fileout, "w")
    finish = f.readline()
    i = 0;
    with open(filename, "r") as f:
        for line in f:
            if i == 0:
                out.write("\\begin{table}[]\n\\centering\n")
                out.write("\\caption{%s}\n" % line)
                out.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n")
                out.write("\\hline\n")
            elif i > 1 and i < 39:
                out.write(" & ".join(line.split(" ")) + ("\\\\ \\hline\n"))
            elif i == 39:
                out.write("\\end{tabular}\n")
                out.write("\\end{table}\n\n")
            i = (i +1) % 45

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: %s <file_in> <file_tables>' % sys.argv[0])
        sys.exit()
    genTable(sys.argv[1], sys.argv[2])
