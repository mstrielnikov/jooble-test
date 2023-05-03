from setuptools import setup, find_packages

setup(
    name="score-standardization-spark",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.2.2",
        "pandas==1.3.5",
        "matplotlib",
        'jupyter',
    ],
    entry_points={
        "console_scripts": [
            "score-standardization-spark = score-standardization-spark.main:main",
        ]
    },
)