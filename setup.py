import setuptools
from setuptools import find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

python_src_path = os.path.join('src')
python_test_path = os.path.join('test')

setuptools.setup(
    name="LammpsAnalysis",
    version="0.0.1",
    author="Physical and Theoretical Chemistry, University of Wuppertal",
    author_email="wissdorf@uni-wuppertal.de",
    description="Analysis / Postprocessing of LAMMPS Simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://team.ipams.uni-wuppertal.de/PTC/lammps_analysis",
    packages=find_packages(python_src_path),
    package_dir={'': python_src_path},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'pandas', 'xarray', 'ipyvolume', 'matplotlib', 'seaborn', 'scipy', 'moviepy']
)
