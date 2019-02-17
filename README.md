# FETCH

fetch is Fast Extragalactic Transient Candidate Hunter.

Install
---

You can use [anaconda](https://www.continuum.io/downloads), and install `fetch` like this:

    conda install -c conda-forge tensorflow-gpu keras scikit-learn pandas scipy numpy matplotlib scikit-image scikit-image scikit-image tqdm 
    git clone https://github.com/devanshkv/fetch.git
    cd fetch
    python setup.py install
    
You would also require `pysigproc` to create the candidate files which can be found [here](https://github.com/devanshkv/pysigproc).

Usage
---
The installation will put `predict.py` and `train.py` in your `PYTHONPATH`.
