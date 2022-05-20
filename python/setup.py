#!/usr/bin/env python

import setuptools 

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='dlav22',
      version='0.01',
      description='Deep Learning for Autonomous Vehicles',
      author='FIXME',
      author_email='FIXME',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/odunkel/dlav_project_olaf',
      package_dir={"": "src"},
      packages=setuptools.find_packages(where="src"),
      python_requires=">=3.7",
     )