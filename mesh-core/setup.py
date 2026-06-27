from setuptools import setup, find_packages

setup(
    name="mesh-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0"
    ]
)
