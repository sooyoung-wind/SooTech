# setup.py

from setuptools import setup, find_packages

setup(
    name='SooTech',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'torch',
    ],
    description='A collection of utility functions by Soo',
    author='Sooyoung Her',
    author_email='sooyoung.wind@gmail.com',
    url='https://github.com/sooyoung-wind/SooTech',
)
