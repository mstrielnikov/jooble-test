from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import stddev, mean, col, udf, lit
from pyspark.sql.types import FloatType
import matplotlib.pyplot as pyplot


# Define function to compute Z-score
def z_score(x: int, mean: int, stddev: int) -> int: 
    return (x - mean) / stddev

# Convert function which compute Z-score into UDF
z_score_udf = udf(z_score, FloatType())

def transform_data(train_df: DataFrame, test_df: DataFrame, cols_names: list[str], cols_suffix : str = "") -> DataFrame:
    """
    Transform data in CSV format by computing the Z-score standardization for each value in selected columns.
    
    Parameters:
        train_df (DataFrame): input DataFrame with train data.
        test_df (DataFrame): input DataFrame with tested data.
        columns_suffix (str): Column header's suffix which will be concat with original column name to create new column name for resulting values
    """

    # Iterate over feature_type_1_{i} columns and perform z-standardization for each value in a feature column
    for col_name in cols_names:
        mean_val = lit(train_df.select(mean(col_name)).collect()[0][0])
        stddev_val = lit(train_df.select(stddev(col_name)).collect()[0][0])

        new_col_name = f"{col_name}{cols_suffix}"

        test_df = test_df.withColumn(col_name, z_score_udf(col(col_name), mean_val, stddev_val))\
                         .withColumnRenamed(col_name, new_col_name)

    return test_df


def plot_dataframe(dataframe: DataFrame, y_col: str, x_col: str, title: str = "") -> pyplot:
    # Select dataframe columns to plot
    pandas_df = dataframe.select(col(y_col), col(x_col)).toPandas()

    # Build the scatter plot
    pyplot.scatter(pandas_df[y_col], pandas_df[x_col])
    pyplot.xlabel(y_col)
    pyplot.ylabel(x_col)
    pyplot.title(title)
    return pyplot


def subplot_dataframes(dataframe: DataFrame, x_col: str, cols_names: list[str]) -> pyplot:
    # Select dataframe columns to plot
    pandas_df = dataframe.select([ col(col_name) for col_name in cols_names ]).toPandas().to_csv()

    number_of_graphs = len(cols_names)
    
    # Create a figure with number_of_graphs subplots arranged in a number_of_graphs x 1 grid
    fig, axs = pyplot.subplots(number_of_graphs, 1)

    for graph_number in range(number_of_graphs):
        axs[graph_number].plot(pandas_df[x_col], pandas_df[graph_number])
        # axs[graph_number].set_tittle()
    
    return pyplot


if __name__ == "__main__":
    output_data_path = "./data/output/test_transformed.csv"

    spark = SparkSession.builder.appName("score-standardization-spark").getOrCreate()

    # Load *.csv
    train_df = spark.read.csv("./data/input/train.csv", header=True, inferSchema=True)
    test_df = spark.read.csv("./data/input/test.csv", header=True, inferSchema=True)

    # Enumerate column headers which starts with `feature_type_1`
    feature_cols = [ header for header in test_df.columns if header.startswith("feature_type_1_") ]

    transformed = transform_data(test_df, train_df, feature_cols, cols_suffix="_stand")

    # Write transformed data to output CSV file with pandas to avoid fragmentation
    transformed.toPandas().to_csv(output_data_path)
    transformed.show()


    plot = subplot_dataframes(transformed, "id", feature_cols)
    plot.show()

    spark.stop()
