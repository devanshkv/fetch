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
The installation will put `predict.py` and `train.py` in your `PYTHONPATH`. To predict a bunch of candidates living in a directory `/data/candidates/` use `predict.py` for model `a` as follows:

        predict.py --data_dir /data/candidates/ --model a
        
To fine-tune the model `a`, with a bunch of canidates, put them in a pandas readable csv, `candidate.csv` with headers 'h5' and 'label'. Use

        train.py --data_csv candidates.csv --model a --output_path ./
        
This would train the model `a` and save the trainig log, and model weights in the output path.
