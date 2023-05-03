import matplotlib.pyplot as plot
import pandas as pandas


"""
def pandas_plot_single(pandas_dataframe: pandas.DataFrame, y_axs: list, x_axs: list, title: str = "") -> plot:

Function to create a scatter plot for a single pair of columns.

Parameters:
    pandas_dataframe (pandas.DataFrame): input data in the form of a Pandas DataFrame.
    y_axs (list): list of column names to be used as the y-axis data.
    x_axs (list): list of column names to be used as the x-axis data.
    title (str): title of the plot.

Returns: plot
"""

def pandas_plot_single(pandas_dataframe: pandas.DataFrame, y_axs: list, x_axs: list, title: str = "") -> plot:
    # Build the scatter plot
    plot.scatter(y_axs, x_axs)
    plot.xlabel(y_axs)
    plot.ylabel(x_axs)
    plot.title(title)
    return plot


"""
def pandas_plot_figure(pandas_dataframe: pandas.DataFrame, col_names: list[str], axs_col_name: str, tittle : str = "") -> plot:

Function to create a plot with multiple lines.

Parameters:
    pandas_dataframe (pandas.DataFrame): input data in the form of a Pandas DataFrame.
    col_names (list): list of column names to be plotted.
    axs_col_name (str): column name to be used as the x-axis data.
    tittle (str): title of the plot.

Returns: plot
"""

def pandas_plot_figure(pandas_dataframe: pandas.DataFrame, col_names: list[str], axs_col_name: str, tittle : str = "") -> plot:
    x_axs = pandas_dataframe[axs_col_name].tolist()
    
    fig, ax = plot.subplots() # Create a matplotlib figure
    for col_name in col_names:
        ax.plot(x_axs, pandas_dataframe[col_name].tolist(), label=col_name)# Set title and labels
    
    ax.set_xlabel(axs_col_name)
    ax.set_ylabel('score')# Add a legend
    ax.legend(loc='lower center', bbox_to_anchor=(1.25, 0.5), ncol=3)
    ax.set_title(tittle)
    return plot
    
    
"""
def pandas_plot_scatter(pandas_dataframe: pandas.DataFrame, col_names: list[str], axs_col_name: str, tittle : str = "") -> plot:

Function to create a scatter plot for multiple pairs of columns.

Parameters:
    pandas_dataframe (pandas.DataFrame): input data in the form of a Pandas DataFrame.
    col_names (list): list of column names to be plotted.
    axs_col_name (str): column name to be used as the x-axis data.
    tittle (str): title of the plot.

Returns: plot
"""

def pandas_plot_scatter(pandas_dataframe: pandas.DataFrame, col_names: list[str], axs_col_name: str, tittle : str = "") -> plot:
    plot.subplots(figsize=(8, 6))
    plot.subplots_adjust(left=0.1)
    plot.title(tittle)
    plot.xlabel(axs_col_name)
    plot.ylabel("z-score")
    
    x_axs = pandas_dataframe[axs_col_name].tolist()
    
    for column in col_names:
        plot.scatter(x_axs, pandas_dataframe[column].tolist(), label=column, s=9)
    
    plot.legend(loc='lower right', fontsize=8)

    return plot