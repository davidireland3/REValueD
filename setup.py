"""Setup script for REValueD package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="revalued",
    version="0.1.0",
    author="David Ireland",
    author_email="david.ireland@warwick.ac.uk",  # Update this
    description="REValueD: Randomised Ensemble Value Decomposition for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidireland3/REValueD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",  # Update if different
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "gymnasium>=0.26.0",
        "dm_control>=1.0.0",
        "pyyaml>=5.4.0",
        "loguru>=0.5.0",
        "wandb>=0.12.0",  # Optional but included
        "dmc-datasets @ git+https://github.com/davidireland3/dmc_datasets.git@main",
    ],
    entry_points={
        "console_scripts": [
            "revalued-train=scripts.train:main",
            "revalued-evaluate=scripts.evaluate:main",
        ],
    },
)
