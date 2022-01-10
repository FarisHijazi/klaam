import setuptools
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = list(filter(None, fh.read().split('\n')))


setuptools.setup(
    name="klaam",
    version="1.0",
    author="Zaid Alyafeai",
    author_email="alyafey22@gmail.com",
    description='Arabic speech recognition and classification using wav2vec models. This repository allows training and prediction using pretrained models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ARBML/klaam",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=requirements,
)
