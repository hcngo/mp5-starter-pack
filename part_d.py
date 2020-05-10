from pyspark.ml.classification import RandomForestClassifier
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

sc = SparkContext()
spark = SparkSession.builder.appName('fun').getOrCreate()

def predict(df_train, df_test):
    # TODO: Train random forest classifier

    # Hint: Column names in the given dataframes need to match the column names
    # expected by the random forest classifier `train` and `transform` functions.
    # Or you can alternatively specify which columns the `train` and `transform`
    # functions should use

    # Result: Result should be a list with the trained model's predictions
    # for all the test data points
    numTrees = 20
    maxDepth = 20
    seed = 0
    rf = RandomForestClassifier(numTrees=numTrees, maxDepth=maxDepth, seed=seed)
    rf_model = rf.fit(df_train)
    predictions = rf_model.transform(df_test)

    results = predictions.rdd.map(lambda r: r.prediction).collect()

    return results

def parse_line_training_data(line):
    # TODO: Parse data from line into an RDD
    # Hint: Look at the data format and columns required by the KMeans fit and
    # transform functions
    tokens = line.split(",")
    label = int(tokens[-1])
    features = Vectors.dense([float(t.strip()) for t in tokens[:-1]])
    return (label, features)

def parse_line_test_data(line):
    tokens = line.split(",")
    label = -1
    features = Vectors.dense([float(t.strip()) for t in tokens])
    return (label, features)

def main():
    raw_training_data = sc.textFile("dataset/training.data")

    # TODO: Convert text file into an RDD which can be converted to a DataFrame
    # Hint: For types and format look at what the format required by the
    # `train` method for the random forest classifier
    # Hint 2: Look at the imports above
    rdd_train = raw_training_data.map(parse_line_training_data)

    # TODO: Create dataframe from the RDD
    df_train = spark.createDataFrame(rdd_train, ["label", "features"])

    raw_test_data = sc.textFile("dataset/test-features.data")

    # TODO: Convert text file lines into an RDD we can use later
    rdd_test = raw_test_data.map(parse_line_test_data)

    # TODO:Create dataframe from RDD
    df_test = spark.createDataFrame(rdd_test, ["label", "features"])

    predictions = predict(df_train, df_test)

    # You can take a look at dataset/test-labels.data to see if your
    # predictions were right
    for pred in predictions:
        print(int(pred))


if __name__ == "__main__":
    main()
