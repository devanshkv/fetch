from setuptools import setup

# The README as the package's description, with its links into the repository made absolute
with open("README.md") as f:
    long_description = f.read().replace("](docs/", "](https://github.com/devanshkv/fetch/blob/tf2/docs/")

setup(
    name="fetch",
    version="0.2.0",
    packages=["fetch", "fetch.pytorch"],
    scripts=["bin/predict.py", "bin/predict_torch.py", "bin/train.py"],
    package_dir={"fetch": "fetch"},
    package_data={"fetch": ["models/model_list.csv", "models/*/*"]},
    url="https://github.com/devanshkv/fetch",
    python_requires=">=3.8",
    install_requires=["numpy>=1.19.5", "pandas>=1.2.2", "h5py>=3.1.0", "scipy>=1.6.0"],
    # predict.py and train.py need tensorflow (train.py also scikit-learn); predict_torch.py needs torch
    extras_require={
        "tensorflow": ["tensorflow[and-cuda]>=2.12", "scikit-learn"],
        "torch": ["torch>=2.0"],
        "test": ["pytest", "pytest-cov"],
    },
    license="GNU General Public License v3.0",
    author="Devansh Agarwal, Kshitij Aggarwal",
    author_email="devansh.kv@gmail.com, ka0064@mix.wvu.edu",
    description="FETCH (Fast Extragalactic Transient Candidate Hunter)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
