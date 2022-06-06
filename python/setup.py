#!/usr/bin/env python

import setuptools 

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='perceptionloomo',
      version='1.0',
      description='Perception Pipeline',
      author='theo hermann',
      author_email='theo.hermann@epfl.ch',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theoh-io/Perception-Pipeline',
      package_dir={"": "src"},
      packages=setuptools.find_packages(where="src"),
      python_requires=">=3.7",
     )