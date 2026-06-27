from setuptools import setup, find_packages

setup(
    name="mesh-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.2.0"
    ]
)
