from setuptools import setup, find_packages

setup(
    name="autoflow-srxn",
    version="0.1.0",
    author="Dongheon",
    description="Automated Surface Reaction (SRXN) Modeling Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/code4simulation/autoflow_SRXN",
    package_dir={"autoflow_srxn": "src"},
    packages=["autoflow_srxn"],
    include_package_data=True,
    package_data={
        "autoflow_srxn": ["*.json"],
    },
    install_requires=[
        "ase",
        "pymatgen",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "mace": ["torch", "torch-geometric", "mace-torch"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)
