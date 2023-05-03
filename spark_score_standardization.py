# Import necessary modules
from src.utils.spark_udf import *
from src.visualization.visualize_plot import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from IPython.display import display

if __name__ == "main":
    SparkAppName = "spark-score-standardization"

    # Define output data path
    output_data_path = "data/output/test_transformed.csv"

    # Create Spark configuration and context
    conf = SparkConf().setAppName(SparkAppName)
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    # Create Spark session
    spark = SparkSession.builder.appName(SparkAppName).getOrCreate()

    # Load train and test data as Spark dataframes from *.csv files
    train_df = spark.read.csv("data/input/train.csv", header=True, inferSchema=True)
    test_df = spark.read.csv("data/input/test.csv", header=True, inferSchema=True)

    # Filter column headers which starts with `feature_type_1_`
    feature_cols = list(filter(lambda header: header.startswith("feature_type_1_"), test_df.columns))

    # Normalize data using Z-score normalization
    normalized_df = normalize_df_z_score(test_df, train_df, feature_cols, cols_new_name_function=lambda col_name : col_name + "_stand")

    # Write the normalized data to a CSV file
    normalized_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_data_path)

    # Convert the Spark dataframe to a Pandas dataframe for visualization
    normalized_df_pandas = normalized_df.toPandas()

    # Filter column headers `feature_type_1_{i}_stand`
    normalized_feature_cols = list(filter(lambda header: header.startswith("feature_type_1_"), normalized_df_pandas.columns))

    # Visualize the normalized data using a scatter plot
    scatter = pandas_plot_scatter(normalized_df_pandas, axs_col_name="id", col_names=normalized_feature_cols, tittle="z-score normaization")
    
    # Save the plot to a file in PNG format
    scatter.savefig("./reports/figures/plot.png")

    # Display the Pandas dataframe
    display(normalized_df_pandas)

    # Stop the Spark session
    spark.stop()
