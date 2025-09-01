from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Comment on utils extra
print("For benchmarking, metrics and utilities, use the [utils] extra.")

# Core dependencies (dire.py and hpindex.py)
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
    "fast-twed",
    "pot"
]

setup(
    name="dire-jax",
    version="0.2.0",
    author="Alexander Kolpakov (UATX), Igor Rivin (Temple University)",
    author_email="akolpakov@uaustin.org, rivin@temple.edu",
    description="A JAX-based Dimension Reducer",
    long_description=long_description,
    long_description_content_type="text/markdown",  
    url="https://github.com/sashakolpakov/dire-jax",
    packages=find_packages(include=["dire_jax", "dire_jax.*", "tests", "tests.*"]),
    include_package_data=True,  
    package_data={"tests": ["*.py", "*.ipynb"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=core_deps,
    extras_require={
        "utils": utils_deps,
    },
)
