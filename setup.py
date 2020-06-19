import os
import re
# To use a consistent encoding
from codecs import open as copen

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with copen(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("bioinformatics_project", "__version__.py")

test_deps =[
    "pytest",
    "pytest-cov",
    "coveralls",
    "validate_version_code",
    "codacy-coverage"
]

extras = {
    'test': test_deps,
}

setup(
    name='bioinformatics_project',
    version=__version__,
    description="Progetto di bioinformatica",
    long_description=long_description,
    url="https://github.com/micheleantonazzi/bioinformatics-project",
    author="micheleantonazzi@gmail.com",
    author_email="micheleantonazzi",
    # Choose your license
    license='MIT',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    # Add here the package dependencies
    install_requires=["tensorflow",
                      "tqdm",
                      "numpy",
                      "pandas",
                      "scikit-learn",
                      "barplots",
                      "plot_keras_history",
                      "encodeproject",
                      "pybwtool",
                      "crr_labels",
                      "epigenomic_dataset",
                      "keras_tqdm",
                      "ucsc_genomes_downloader",
                      "keras_bed_sequence",
                      "cache_decorator",
                      "encodeproject",
                      "pybwtool",
                      "ray",
                      "termcolor",
                      "minepy",
                      "seaborn",
                      "Boruta",
                      "prince",
                      "pickle-mixin",
                      "tensorflow_addons"],
    extras_require=extras,
)
