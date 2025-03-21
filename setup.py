from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Comment on utils extra
print("For benchmarking, metrics and utilities, use [utils] extra.")

# Core dependencies (dire.py)
core_deps = [
    "jax",
    "numpy",
    "scipy",
    "tqdm",
    "pandas",
    "plotly",
    "loguru",
    "scikit-learn"
]

# Dependencies for utils and metrics (dire_utils.py and hpmetrics.py)
utils_deps = [
    "ripser",
    "persim",
    "fastdtw",
    "pytwed",
    "pot"
]

setup(
    name="dire-jax",
    version="0.0.1",
    author="Alexander Kolpakov, Igor Rivin",
    author_email="akolpakov@uaustin.org, rivin@temple.edu",
    description="A JAX-based Dimension Reducer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sashakolpakov/dire-jax",
    packages=find_packages(include=["dire_jax", "dire_jax.*", "tests", "tests.*"]),
    include_package_data=True,  # Ensures non-Python files (like test data) are included
    package_data={"tests": ["*.py", "*.ipynb"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=core_deps,
    extras_require={
        "utils": utils_deps,
    },
)
