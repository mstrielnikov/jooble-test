from setuptools import setup, find_packages

setup(
    name="jooble-test",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.2.0",
        "pandas==1.3.5",
    ],
    entry_points={
        "console_scripts": [
            "jooble-test = jooble_test.main:main",
        ]
    },
)
