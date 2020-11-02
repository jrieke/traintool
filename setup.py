from pathlib import Path
from setuptools import setup, find_packages

parent_dir = Path(__file__).resolve().parent

setup(
    name="traintool",
    version=parent_dir.joinpath("traintool/_version.txt").read_text(encoding="utf-8"),
    author="Johannes Rieke",
    author_email="johannes.rieke@gmail.com",
    description="Train pre-implemented machine learning models with one line of code",
    long_description=parent_dir.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/jrieke/traintool",
    license="",
    packages=find_packages(exclude=("tests", "docs", "examples")),
    package_data={"": ["_version.txt"]},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "pytorch-ignite",
        "comet_ml",
        "pyyaml>=5.1",  # for sort_keys arg
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "joblib",
        "tensorboardX",
        "tensorboard",
        "imageio",
        "loguru",
        "editdistance",
    ],
    entry_points="""""",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
    ],
)
