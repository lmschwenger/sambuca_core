from setuptools import setup, find_packages
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sambuca', 'core'))
from sambuca.core import __version__

setup(
    name="sambuca-core",
    version=__version__,
    author="Lasse M. Schwenger",
    author_email="lasse.m.schwenger@gmail.com",
    description="A short description of sambuca_core",
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