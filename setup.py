from setuptools import setup, find_packages

setup(
    name="bp-xray-research",
    version="0.1.0",
    description="Data science suite for Bardeen-Petterson X-ray polarization research",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "torch>=2.0",
        "pyro-ppl>=1.8",
        "astropy>=5.3",
        "h5py>=3.9",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    python_requires=">=3.10",
)
