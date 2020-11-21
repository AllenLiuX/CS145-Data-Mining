Information about code files:

*DataPoints.py
	Helper class/functions.
	The different statistical functions (mean, stddev, covariance matrix) are implemented to be used by GMM.
	There are also functions to properly analyze and print DataPoints as output.
	
*KMeans.py
	1. The file prints results from the algorithm run on all three datasets.
	2. Points with cluster labels are stored in KMeans.csv (currently stores the result of dataset 3; change the last dataset run for other dataset results).
	3. KMeans initializes centroids, and performs clustering to the closest centroid by Euclidean distance. This is repeated until no more changes are made.

*DBSCAN.py
	1. The file prints results from the algorithm run on all three datasets.
	2. Points with cluster labels are stored in DBSCAN_dataset3.csv (currently stores the result of dataset 3; change the last dataset run for other dataset results).
	3. DBSCAN adds all the points in the Eps neighborhood (> minPts) to a cluster, and iterates through all points to add all density reachable points.
	4. If the neighborhood has < minPts, it is labelled as noise.

*GMM.py
	1. The file prints results from the algorithm run on all three datasets.
	2. Points with cluster labels are stored in GMM.csv (currently stores the result of dataset 3; change the last dataset run for other dataset results).
	3. The updates are performed based on the probabilities of the different <class, cluster> pairs. The EM-step use parameters such as W[i][j] (i.e. the probabilities), and statistical features such as mean, stddev, and covariance to iteratively update the value of the weights and cluster the datapoints.
	4. Give this some time to run, as it is a computationally expensive program.
