"""Installation with setuptools or pip."""
from setuptools import setup, find_packages
import os
import ast


def get_version_from_init():
    """Obtain library version from main init."""
    init_file = os.path.join(os.path.dirname(__file__), "failsim", "__init__.py")
    with open(init_file) as fd:
        for line in fd:
            if line.startswith("__version__"):
                return ast.literal_eval(line.split("=", 1)[1].strip())


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    lic = f.read()


setup(
    name="failsim",
    version=get_version_from_init(),
    description="LHC Fast failure simulation toolkit",
    long_description=readme,
    author="Cédric Hernaslteens, Oskari Tuormaa",
    author_email="cedric.hernalsteens@cern.ch, oskari.kristian.tuormaa@cern.ch",
    url="https://gitlab.cern.ch/machine-protection/fast-beam-failures",
    license=lic,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    install_requires=[
        "numpy",
        "pandas",
        "pymask",
        "pyyaml",
        "plotly",
    ],
    package_data={"failsim": ["data"]},
)
