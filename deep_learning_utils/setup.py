from setuptools import setup, find_packages

setup(
    name="deep_learning_utils",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
