from setuptools import setup, find_packages

setup(
    name="spark-score-standardization",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.2.2",
        "pandas==1.3.5",
        "matplotlib",
        "jupyter",
        "glob",
        "nbconvert"
    ],
    entry_points={
        "console_scripts": [
            "spark-score-standardization = spark-score-standardization.main:main",
        ]
    },
)