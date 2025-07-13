import os
import sys

from setuptools import setup, find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sambuca', 'core'))
from sambuca.core import __version__

setup(
    name="sambuca-core",
    version=__version__,
    author="Lasse M. Schwenger",
    author_email="lasse.m.schwenger@gmail.com",
    description="Implementation of the SAMBUCA model (Semi-Analytical Model for Bathymetry, Un-mixing, and Concentration Assessment) developed by CSIRO for remote sensing applications.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/lmschwenger/sambuca_core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "tqdm>=4.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
)
