# setup.py
from setuptools import setup, find_packages

setup(
    name='repoprompter',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'PyGithub',
    ],
    entry_points={
        'console_scripts': [
            'repoprompter=repoprompter.repoprompter.main:main',
        ],
    },
)
