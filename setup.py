from setuptools import setup

setup(
    name="fetch",
    version="0.2.0",
    packages=["fetch"],
    scripts=["bin/predict.py", "bin/train.py"],
    package_dir={"fetch": "fetch"},
    package_data={"fetch": ["models/model_list.csv", "models/*/*yaml"]},
    url="https://github.com/devanshkv/fetch",
    tests_require=["pytest", "pytest-cov"],
    license="GNU General Public License v3.0",
    author=["Devansh Agarwal", "Kshitij Aggarwal"],
    author_email=["devansh.kv@gmail.com", "ka0064@mix.wvu.edu"],
    description="FETCH (Fast Extragalactic Transient Candidate Hunter)",
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
