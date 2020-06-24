# FETCH

[![DOI](https://zenodo.org/badge/165734093.svg?style=flat-square)](https://zenodo.org/badge/latestdoi/165734093)
[![issues](https://img.shields.io/github/issues/devanshkv/fetch)](https://github.com/devanshkv/fetch/issues)
[![forks](https://img.shields.io/github/forks/devanshkv/fetch)](https://github.com/devanshkv/fetch/network/members)
[![stars](https://img.shields.io/github/stars/devanshkv/fetch)](https://github.com/devanshkv/fetch/stargazers)
[![GitHub license](https://img.shields.io/github/license/devanshkv/fetch)](https://github.com/devanshkv/fetch/blob/master/LICENSE)
[![HitCount](http://hits.dwyl.com/devanshkv/fetch.svg)](http://hits.dwyl.com/devanshkv/fetch)
[![arXiv](https://img.shields.io/badge/arXiv-1902.06343-brightgreen.svg)](https://arxiv.org/abs/1902.06343)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fdevanshkv%2Ffetch)]()



fetch is Fast Extragalactic Transient Candidate Hunter. It has been detailed in the paper [Towards deeper neural networks for Fast Radio Burst detection](https://arxiv.org/abs/1902.06343)

Install
---

We suggest using [anaconda](https://www.continuum.io/downloads) for using FETCH.

First we need to install cudatoolkit matching the installed cuda version.

For cuda 8.0 `conda install -c anaconda cudatoolkit==8.0 tensorflow-gpu==1.4.1`

For cuda 9.0 `conda install -c anaconda cudatoolkit==9.0 tensorflow-gpu==1.12.0`

For cuda 9.2 `conda install -c anaconda cudatoolkit==9.2 tensorflow-gpu==1.12.0`

For cuda 10. `conda install -c anaconda cudatoolkit==10.0.130 tensorflow-gpu==1.13.1`

__Note__: `tensorflow` installation from `conda-forge` channel does not work with GPUs.

You would also require `pysigproc` to create the candidate files which can be found [here](https://github.com/devanshkv/pysigproc).


Now we can install `fetch` like this:

    conda install -c anaconda keras scikit-learn pandas scipy numpy matplotlib scikit-image tqdm numba pyyaml=3.13
    git clone https://github.com/devanshkv/fetch.git
    cd fetch
    python setup.py install

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

Example
---

Test filterbank data can be downloaded from [here](http://astro.phys.wvu.edu/files/askap_frb_180417.tgz). The folder contains three filterbanks: 28.fil  29.fil  34.fil.
Heimdall results for each of the files are as follows:

for 28.fil

    16.8128	1602	2.02888	1	127	475.284	22	1601	1604
for 29.fil

    18.6647	1602	2.02888	1	127	475.284	16	1601	1604
for 34.fil

    13.9271	1602	2.02888	1	127	475.284	12	1602	1604 

The `cand.csv` would look like the following:

    28.fil,16.8128,2.02888,475.284,1
    29.fil,18.6647,2.02888,475.284,1
    34.fil,13.9271,2.02888,475.284,1
    
Running `candmaker.py` will create three files:

    cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_13.92710.h5
    cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_16.81280.h5
    cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_18.66470.h5

Running `predict.py` with model `a` will give `results_a.csv`:

    ,candidate,probability,label
    0,cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_18.66470.h5,1.0,1.0
    1,cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_16.81280.h5,1.0,1.0
    2,cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_13.92710.h5,1.0,1.0
    
    
Training Data
---

The training data is available at [astro.phys.wvu.edu/fetch](http://astro.phys.wvu.edu/fetch/).
