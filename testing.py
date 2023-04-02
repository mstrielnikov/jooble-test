from pyspark.sql import SparkSession
from pyspark.sql.functions import stddev, mean, col, udf, lit
from pyspark.sql.types import FloatType
from os import getcwd

spark = SparkSession.builder.appName("jooble-test").getOrCreate()

# Define function to compute Z-score
def z_score(x, mean, stddev): 
    return (x - mean) / stddev

# Convert function which compute Z-score into UDF
z_score_udf = udf(z_score, FloatType())

def transform_data(input_test_path, input_train_path, output_path):
    """
    Transform data in CSV format by computing the Z-score for each value in selected columns.
    
    Parameters:
        input_test_path (str): Path to the input test CSV file.
        input_train_path (str): Path to the input CSV file with training data.
        output_path (str): Path to the output CSV file.
    """
    # Load *.csv
    train_df = spark.read.csv(input_train_path, header=True, inferSchema=True)
    test_df = spark.read.csv(input_test_path, header=True, inferSchema=True)

    # Enumerate column headers which starts with `feature_type_1_`
    feature_cols = [ header for header in test_df.columns if header.startswith('feature_type_1_') ]

    # Iterate over feature_type_1_{i} columns and perform z-standardization for each value in a feature column
    for feature_col in feature_cols:
        mean_val = lit(train_df.select(mean(feature_col)).collect()[0][0])
        stddev_val = lit(train_df.select(stddev(feature_col)).collect()[0][0])

        new_column_name = "feature_type_1_stand_{}".format(feature_col.removeprefix('feature_type_1_'))

        test_df = test_df.withColumn(feature_col, z_score_udf(col(feature_col), mean_val, stddev_val))\
                         .withColumnRenamed(feature_col, new_column_name)

    # Write transformed data to output CSV file with pandas to avoid fragmentation
    test_df.toPandas().to_csv(output_path)

    return test_df


if __name__ == "__main__":
    WorkDir = getcwd()
    train_data_path = f"{WorkDir}/data/input/train.csv"
    test_data_path = f"{WorkDir}/data/input/test.csv"
    output_data_path = f"{WorkDir}/data/output/test_transformed.csv"
    transformed = transform_data(test_data_path, train_data_path, output_data_path)
    transformed.show()
    spark.stop()
