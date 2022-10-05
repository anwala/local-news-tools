#!/usr/bin/env python

from setuptools import setup, find_packages
from locnews.config import __appversion__
desc = """Local News Tools: Studying the United States via Local Media"""

setup(
    name='locnews',
    version=__appversion__,
    description=desc,
    long_description='See: https://github.iu.edu/anwala/us-pulse',
    author='Alexander C. Nwala',
    author_email='anwala@iu.edu',
    url='https://github.iu.edu/anwala/us-pulse',
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    package_data={
        'locnews': [
            './Resources/*',
            './Resources/*/*',
            './Resources/*/*/*'
        ]
    },
    install_requires=[
        'NwalaTextUtils @ git+git://github.com/oduwsdl/NwalaTextUtils.git',
        'sgsuite @ git+git://github.com/oduwsdl/storygraph-suite.git',
        'sklearn'
    ],
    scripts=[
        'bin/locnews'
    ]
)
