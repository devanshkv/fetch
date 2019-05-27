# FETCH

fetch is Fast Extragalactic Transient Candidate Hunter. It has been detailed in the paper [Towards deeper neural networks for Fast Radio Burst detection](https://arxiv.org/abs/1902.06343)

Install
---

We suggest using [anaconda](https://www.continuum.io/downloads) for using FETCH.

First we need to install cudatoolkit matching the installed cuda version.

For cuda 7.5 `conda install cudatoolkit==7.5`

For cuda 8.0 `conda install cudatoolkit==8.0`

For cuda 9.0 `conda install cudatoolkit==9.0`

For cuda 9.2 `conda install cudatoolkit==9.2`

For cuda 10. `conda install cudatoolkit==10.0.130`

__Note__: `tensorflow` installation from `conda-forge` channel does not work with GPUs.

Now we can install `fetch` like this:

    conda install -c anaconda tensorflow-gpu keras scikit-learn pandas scipy numpy matplotlib scikit-image tqdm numba pyyaml=3.13
    git clone https://github.com/devanshkv/fetch.git
    cd fetch
    python setup.py install
    
You would also require `pysigproc` to create the candidate files which can be found [here](https://github.com/devanshkv/pysigproc).

The installation will put `predict.py`,`candmaker.py` and `train.py` in your `PYTHONPATH`.

Usage
---
First create a candidate file (`cands.csv`) of the following format:

        /path/to/filterbank/myfilterbank.fil,S/N,start_time,dm,boxcar_width,label,path_to_kill_mask
       
here `boxcar_width` is in units of `int(log2(number of samples))`. `path_to_kill_mask` is a numpy readable file with channel numbers to kill. If not required, this field can be left empty.

Next, to generate the candidate files containing DM-time and Frequency-time arrays for classification use `candmaker.py`. Saving candidate h5s with their parameters in `cands.csv` to a directory `/data/canddidates/` and rebinning the time and frequency axis to 256 bins using decimation can be done by: 

        candmaker.py --frequency_size 256 --time_size 256 --cand_param_file cands.csv --plot --fout /data/candidates/
       
To predict a these candidate h5 files living in the directory `/data/candidates/` use `predict.py` for model `a` as follows:

        predict.py --data_dir /data/candidates/ --model a
        
To fine-tune the model `a`, with a bunch of candidates, put them in a pandas readable csv, `candidate.csv` with headers 'h5' and 'label'. Use

        train.py --data_csv candidates.csv --model a --output_path ./
        
This would train the model `a` and save the training log, and model weights in the output path.
