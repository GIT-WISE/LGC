from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="lgc-molnet",
    version="0.1.0",
    description="Cayley-Legendre graph convolution model for MoleculeNet property prediction",
    packages=find_packages(include=["lgc_molnet", "lgc_molnet.*"]),
    install_requires=requirements,
    python_requires=">=3.9",
)
