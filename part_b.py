from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as F

############################################
#### PLEASE USE THE GIVEN PARAMETERS     ###
#### FOR TRAINING YOUR KMEANS CLUSTERING ###
#### MODEL                               ###
############################################

NUM_CLUSTERS = 4
SEED = 0
MAX_ITERATIONS = 100
INITIALIZATION_MODE = "random"

sc = SparkContext()
spark = SparkSession.builder.appName('fun').getOrCreate()


def get_clusters(df, num_clusters, max_iterations, initialization_mode,
                 seed):
    # TODO:
    # Use the given data and the cluster pparameters to train a K-Means model
    # Find the cluster id corresponding to data point (a car)
    # Return a list of lists of the titles which belong to the same cluster
    # For example, if the output is [["Mercedes", "Audi"], ["Honda", "Hyundai"]]
    # Then "Mercedes" and "Audi" should have the same cluster id, and "Honda" and
    # "Hyundai" should have the same cluster id
    # return [[]]
    kmeans = KMeans(k=num_clusters, seed=seed, maxIter=max_iterations, initMode=INITIALIZATION_MODE)
    model = kmeans.fit(df)
    results = model.transform(df)
    clusters = {}
    for record in results.rdd.collect():
        if not(record.prediction in clusters):
            clusters[record.prediction] = []
        clusters[record.prediction].append(record.id)

    return clusters.values()


def parse_line(line):
    # TODO: Parse data from line into an RDD
    # Hint: Look at the data format and columns required by the KMeans fit and
    # transform functions
    tokens = line.split(",")
    id = tokens[0]
    features = Vectors.dense([float(t.strip()) for t in tokens[1:]])
    return (id, features)


if __name__ == "__main__":
    f = sc.textFile("dataset/cars.data")

    rdd = f.map(parse_line)

    # TODO: Convert RDD into a dataframe
    df = spark.createDataFrame(rdd, ["id", "features"])

    clusters = get_clusters(df, NUM_CLUSTERS, MAX_ITERATIONS,
                            INITIALIZATION_MODE, SEED)
    for cluster in clusters:
        print(','.join(cluster))
