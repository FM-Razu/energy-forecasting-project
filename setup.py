from setuptools import setup, find_packages

setup(
    name="ml-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "fastapi",
        "uvicorn",
        "streamlit",
        "mlflow",
        "great-expectations",
        "pytest",
        "plotly",
    ],
)