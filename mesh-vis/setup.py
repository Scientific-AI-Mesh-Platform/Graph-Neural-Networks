from setuptools import setup, find_packages

setup(
    name="mesh-vis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "matplotlib>=3.5.0",
        "numpy>=1.20.0"
    ]
)
