from setuptools import setup, find_packages

setup(
    name="mesh-sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.2.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0"
    ]
)
