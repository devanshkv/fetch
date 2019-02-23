# FETCH

fetch is Fast Extragalactic Transient Candidate Hunter. It has been detailed in the paper [Towards deeper neural networks for Fast Radio Burst detection](https://arxiv.org/abs/1902.06343)

Install
---

You can use [anaconda](https://www.continuum.io/downloads), and install `fetch` like this:

    conda install -c anaconda tensorflow-gpu keras scikit-learn pandas scipy numpy matplotlib scikit-image scikit-image scikit-image tqdm 
    git clone https://github.com/devanshkv/fetch.git
    cd fetch
    python setup.py install
    
You would also require `pysigproc` to create the candidate files which can be found [here](https://github.com/devanshkv/pysigproc).

Usage
---
The installation will put `predict.py` and `train.py` in your `PYTHONPATH`. To predict a bunch of candidate h5 files living in a directory `/data/candidates/` use `predict.py` for model `a` as follows:

        predict.py --data_dir /data/candidates/ --model a
        
To fine-tune the model `a`, with a bunch of candidates, put them in a pandas readable csv, `candidate.csv` with headers 'h5' and 'label'. Use

        train.py --data_csv candidates.csv --model a --output_path ./
        
This would train the model `a` and save the training log, and model weights in the output path.

To generate the candidate h5 files containing DM-time and Frequency-time arrays, `candmaker.py` can be used. It can also be used to rebin the candidates on the fly. For example: Saving candidate h5s with their parameters in `cands.csv` to a directory `/my/canddidates/` and rebinning the time and frequency axis to 256 bins using decimation can be done by: 

        candmaker.py --frequency_size 256 --time_size 256 --cand_param_file cands.csv --plot --fout /my/candidates/
        
A typical example of `cands.csv` file would be (here `boxcar_width` is in units of log2(number of samples)):
        
        /path/to/filterbank/myfilterbank.fil,S/N,start_time,dm,boxcar_width,label
