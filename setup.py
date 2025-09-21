from pathlib import Path
from setuptools import setup, find_packages

# Comment on utils extra (only when running setup.py directly)
import sys
if 'setup.py' in sys.argv[0]:
    print("For benchmarking, metrics and utilities, use the [utils] extra.")

# Only define configuration if pyproject.toml is not available or being used
# This prevents conflicts when both files exist
try:
    # Check if pyproject.toml exists and has project configuration
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # fallback for older Python versions

    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)

    # If pyproject.toml has project config, use minimal setup.py
    if "project" in pyproject_data:
        setup()
    else:
        raise FileNotFoundError  # Fall through to full setup

except (ImportError, FileNotFoundError, Exception):
    # Fallback to full setup.py configuration for compatibility
    # Read the long description from README.md
    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")

    # Core dependencies (dire.py and hpindex.py)
    core_deps = [
        "jax",
        "jaxlib",
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
        version="0.2.1",
        author="Alexander Kolpakov, Igor Rivin",
        author_email="akolpakov@uaustin.org, rivin@temple.edu",
        description="A JAX-based Dimension Reducer",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/sashakolpakov/dire-jax",
        packages=find_packages(exclude=["benchmarking*"]),
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        install_requires=core_deps,
        extras_require={
            "utils": utils_deps,
            "all": utils_deps,
        },
    )
