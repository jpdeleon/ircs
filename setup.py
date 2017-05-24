#!/usr/bin/env python
import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def rd(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

setup(
    name='ircs_pol',
    packages =['ircs'],
    version="0.1.1",
    author='Jerome de Leon',
    author_email = 'jpdeleon.bsap@gmail.com',
    url = 'https://github.com/jpdeleon/ircs',
    license = ['GNU GPLv3'],
    description ='A simple data reduction pipeline for Subaru IRCS linear polarimetry data.',
    long_description=rd("README.md") + "\n\n"
                    + "---------\n\n",
    package_dir={"ircs": "ircs"},

    scripts=['scripts/ircs-imaging', 'scripts/ircs-polarimetry', 'scripts/ircs-analysis', 'scripts/show_raw_image', 'scripts/image_sorter'],
    include_package_data=True,
    keywords=['IRCS','linear polarimetry','near-infrared'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
        ],
    install_requires = ['numpy', 'matplotlib', 'astropy', 'scipy', 'photutils', 'pandas', 'tqdm','scikit-image', 'pyraf'] #, 'pyraf'
)
