from setuptools import setup, find_packages

setup(
    name="apps-eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "numpy",
        "typing_extensions",
    ],
    python_requires=">=3.10",
    description="Evaluation framework for APPS coding problems",
    author="Julian Stastny",
    author_email="",  # Add your email if you want
    url="https://github.com/julianstastny/apps-eval",  # Add your repo URL if you want
) 