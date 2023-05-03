from pyspark.sql import DataFrame
from pyspark.sql.functions import stddev, mean, col, udf, lit
from pyspark.sql.types import FloatType

"""
Args:
----------
x: The value to compute the Z-score.
mean: The mean value for the column.
stddev: The standard deviation value for the column.

Returns:
----------
The Z-score computed for the given x value, mean, and standard deviation.

"""

def z_score(x: int, mean: int, stddev: int) -> int:    
    try:
        z_score = (x - mean) / stddev
    except ZeroDivisionError as e:
        raise(e)
    else:
        return z_score


# Convert function which compute Z-score into UDF
z_score_udf = udf(z_score, FloatType())

"""
Parameters:
----------
train_df : PySpark DataFrame
    Input DataFrame with train data.
test_df : PySpark DataFrame
    Input DataFrame with tested data.
cols_names : list[str]
    Column header's suffix which will be concat with original column name to create new column name for resulting values
cols_new_name_function : callable, optional
    An optional function to modify the column header of each resulting column.

Returns:
----------
PySpark DataFrame
    A new DataFrame containing Z-score normalized data.
"""

# Iterate over feature_type_1_{i} columns and perform z-standardization for each value in a feature column
def normalize_df_z_score(train_df: DataFrame, test_df: DataFrame, cols_names: list[str], cols_new_name_function : callable = lambda col_new_name : col_new_name) -> DataFrame:    
    for col_name in cols_names:
        mean_val = lit(train_df.select(mean(col_name)).collect()[0][0])
        stddev_val = lit(train_df.select(stddev(col_name)).collect()[0][0])

        new_col_name = cols_new_name_function(col_name)

        test_df = test_df.withColumn(col_name, z_score_udf(col(col_name), mean_val, stddev_val))\
                         .withColumnRenamed(col_name, new_col_name)

    return test_df
