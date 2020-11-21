import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def make_graphs(filename):
    x_values = []
    y_values = []
    class_values = []
    with open(filename, 'rb') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        for row in data_reader:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))
            class_values.append(int(row[2]))
    plt.scatter(x_values, y_values, c=class_values)
    plt.show()

if __name__=="__main__":
    if len(sys.argv) != 2:
        pass
    make_graphs(sys.argv[1])
