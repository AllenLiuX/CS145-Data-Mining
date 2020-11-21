# =======================================================================
import sys
import math
# =======================================================================
class DataPoints:
    # -------------------------------------------------------------------
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.isAssignedToCluster = False
    # -------------------------------------------------------------------
    def __key(self):
        return (self.label, self.x, self.y)
    # -------------------------------------------------------------------
    def __eq__(self, other):
        return self.__key() == other.__key()
    # -------------------------------------------------------------------
    def __hash__(self):
        return hash(self.__key())
    # -------------------------------------------------------------------
    @staticmethod
    def getMean(clusters, mean):
        # Initialize the mean for each cluster
        # ****************Please Fill Missing Lines Here*****************
        # iterate through each cluster with an index, num
        for num, cluster in enumerate(clusters):
            x_mean = 0.0
            y_mean = 0.0
            # separately sum x and y values of all points in cluster
            for point in cluster:
                x_mean += point.x
                y_mean += point.y
            # find mean as sum / length, assign [0] to x_mean and [1] to y_mean
            mean[num][0] = x_mean / float(len(cluster))
            mean[num][1] = y_mean / float(len(cluster))

    # -------------------------------------------------------------------
    @staticmethod
    def getStdDeviation(clusters, mean, stddev):
        # Initialize the std for each cluster
        # ****************Please Fill Missing Lines Here*****************
        # iterate through each cluster with an index, num
        for num, cluster in enumerate(clusters):
            x_std = 0.0
            y_std = 0.0
            # for each point, add (x - mean_x) / length to x_std, and same for y
            for point in cluster:
                x_std += (point.x - mean[num][0]) * (point.x - mean[num][0]) / float(len(cluster))
                y_std += (point.y - mean[num][1]) * (point.y - mean[num][1]) / float(len(cluster))
            # standard deviation is the root of previously calculated sum
            stddev[num][0] = math.sqrt(x_std)
            stddev[num][1] = math.sqrt(y_std)

    # -------------------------------------------------------------------
    @staticmethod
    def getCovariance(clusters, mean, stddev, cov):
        # Initialize the cov for each cluster
        # ****************Please Fill Missing Lines Here*****************
        # iterate through each cluster with an index, num
        for num, cluster in enumerate(clusters):
            # cov(X,X), cov(Y,Y) are var(X) = stddev(X) ^ 2 and var(y) respectively
            cov[num][0][0] = math.pow(stddev[num][0], 2)
            cov[num][1][1] = math.pow(stddev[num][1], 2)
            cov_xy = 0.0
            # for each point in cluster, add (x - mean_x) * (y - mean_y) / length to get cov(X,Y)
            for point in cluster:
                cov_xy += (point.x - mean[num][0]) * (point.y - mean[num][1]) / float(len(cluster))
            # cov(X,Y) = cov(Y,X)
            cov[num][1][0] = cov_xy
            cov[num][0][1] = cov_xy

    # -------------------------------------------------------------------
    @staticmethod
    def getNMIMatrix(clusters, noOfLabels):
        nmiMatrix = [[0 for x in range(len(clusters) + 1)] for y in range(noOfLabels + 1)]
        clusterNo = 0
        for cluster in clusters:
            labelCounts = {}
            for point in cluster:
                if not point.label in labelCounts:
                    labelCounts[point.label] = 0
                labelCounts[point.label] += 1
            max = sys.maxsize
            labelNo = 0
            labelTotal = 0
            labelCounts_sorted = sorted(labelCounts.items(), key=lambda x: (x[1], x[0]), reverse=True)
            for label, val in labelCounts_sorted:
                nmiMatrix[label - 1][clusterNo] = labelCounts[label]
                labelTotal += labelCounts.get(label)
            nmiMatrix[noOfLabels][clusterNo] = labelTotal
            clusterNo += 1
            labelCounts.clear()

        # populate last col
        lastRowCol = 0
        for i in range(len(nmiMatrix) - 1):
            totalRow = 0
            for j in range(len(nmiMatrix[i]) - 1):
                totalRow += nmiMatrix[i][j]
            lastRowCol += totalRow
            nmiMatrix[i][len(clusters)] = totalRow
        nmiMatrix[noOfLabels][len(clusters)] = lastRowCol
        return nmiMatrix
    # -------------------------------------------------------------------
    @staticmethod
    def calcNMI(nmiMatrix):
        # calculate I
        row = len(nmiMatrix)
        col = len(nmiMatrix[0])
        N = nmiMatrix[row - 1][col - 1]
        I = 0.0
        HOmega = 0.0
        HC = 0.0
        for i in range(row - 1):
            for j in range(col - 1):
                logPart = (float(N) * nmiMatrix[i][j]) / (float(nmiMatrix[i][col - 1]) * nmiMatrix[row - 1][j])
                if logPart == 0.0:
                    continue
                I += (nmiMatrix[i][j] / float(N)) * math.log(float(logPart))
                logPart1 = nmiMatrix[row - 1][j] / float(N)
                if logPart1 == 0.0:
                    continue
                HC += nmiMatrix[row - 1][j] / float(N) * math.log(float(logPart1))
            HOmega += nmiMatrix[i][col - 1] / float(N) * math.log(nmiMatrix[i][col - 1] / float(N))

        return I / math.sqrt(HC * HOmega)
    # -------------------------------------------------------------------
    @staticmethod
    def getNoOFLabels(dataSet):
        labels = set()
        for point in dataSet:
            labels.add(point.label)
        return len(labels)
    # -------------------------------------------------------------------
    @staticmethod
    def writeToFile(noise, clusters, fileName):
        # write clusters to file for plotting
        f = open(fileName, 'w')
        for pt in noise:
            f.write(str(pt.x) + "," + str(pt.y) + ",0" + "\n")
        for w in range(len(clusters)):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str((w + 1)) + "\n")
        f.close()
# =======================================================================
