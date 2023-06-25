from setuptools import setup, find_packages

setup(
    name='NLP Training Script',
    version='1.0.0',
    description='A script for training an AI using an NLP model',
    author='Matthew Ford',
    author_email='matthew@symbiotic.love',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pandas==2.0.1',
        'scikit-learn==1.2.2',
        'tensorflow==2.12.0'
    ],
)
