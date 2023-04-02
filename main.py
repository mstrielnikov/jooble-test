from pyspark.sql import SparkSession
from pyspark.sql.functions import stddev, mean, col, udf, lit
from pyspark.sql.types import DoubleType, FloatType, IntegerType, ArrayType
from pyspark.ml.feature import StandardScaler
from os import getcwd

WorkDir = getcwd()

spark = SparkSession.builder.appName("jooble-test").getOrCreate()

# Define function to compute Z-score
def z_score(x, mean, stddev): 
    return (x - mean) / stddev

# Convert function which compute Z-score into UDF
z_score_udf = udf(z_score, FloatType())

# Load *.csv
train_df = spark.read.csv(f"{WorkDir}/data/input/train.csv", header=True)
test_df = spark.read.csv(f"{WorkDir}/data/input/test.csv", header=True)

# Enumerate column headers which starts with `feature_type_1_`
feature_cols = [ header for header in test_df.columns if header.startswith('feature_type_1_') ]

# Compute mean and standard deviation for each `feature_type_1`-like column from train.csv
means = train_df.select([ mean(col(column)) for column in feature_cols ])
stddevs = train_df.select([ stddev(col(column)) for column in feature_cols ])

# Iterate over feature_type_1_{i} columns and perform z-standardization for each value in a feature column
for feature_col in feature_cols:
    mean_val = means.select(f"avg({feature_col})").collect()[0][0]
    stddev_val = stddevs.select(f"stddev_samp({feature_col})").collect()[0][0]

    mean_lit = lit(mean_val).cast(FloatType())
    stddev_lit = lit(stddev_val).cast(FloatType())

    test_df = test_df.withColumn(feature_col, z_score_udf(col(feature_col).cast(FloatType()), mean_lit, stddev_lit))
    
test_df.show()

# Df.write.csv(f"{WorkDir}/data/output/test_transformed.csv", header=True)

spark.stop()


