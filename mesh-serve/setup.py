from setuptools import setup, find_packages

setup(
    name="mesh-serve",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "streamlit",
        "matplotlib",
        "mesh-sim",
        "mesh-ml",
        "mesh-vis"
    ]
)
