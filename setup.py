from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.0'
DESCRIPTION = 'Processing of hyperspectral data.'
NAME = "hyperspec"

# Setting up
setup(
    name=NAME,
    version=VERSION,
    author="Yannick Zander",
    author_email="yzander@marum.de",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/yaza11/hyperspec",
    packages=find_packages(),
    install_requires=['matplotlib',
                      'numpy>=2.0.0',
                      'tqdm',
                      'scikit-image',
                      'spectral',
                      'h5py'],
    extras_require={'dev': 'twine'},
    keywords=['python', 'hyperspectral imaging'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)