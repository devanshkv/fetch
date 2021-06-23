#!/usr/bin/env python3

import argparse
import logging
import pathlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pysigproc import SigprocFile
from candidate import Candidate, crop
from h5plotter import plot_h5

logger = logging.getLogger()


def normalise(data):
    """
    Noramlise the data by unit standard deviation and zero median
    :param data: data
    :return:
    """
    data = np.array(data, dtype=np.float32)
    data -= np.median(data)
    data /= np.std(data)
    return data


def cand2h5(cand_val):
    """
    TODO: Add option to use cand.resize for reshaping FT and DMT
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :param cand_val: List of candidate parameters (fil_name, snr, width, dm, label, tcand(s))
    :type cand_val: Candidate
    :return: None
    """
    fil_name, snr, width, dm, label, tcand, kill_mask_path, args = cand_val
    #get the time of the pulse
    tstart = fil_name.find('pow_')
    tend = fil_name.find('_sb')
    pulse_t = fil_name[tstart:tend-3]
    if kill_mask_path == kill_mask_path:
        kill_mask_file = pathlib.Path(kill_mask_path)
        if kill_mask_file.is_file():
            logging.info(f'Using mask {kill_mask_path}')
            kill_chans = np.loadtxt(kill_mask_path, dtype=np.int)
            filobj = SigprocFile(fil_name)
            kill_mask = np.zeros(filobj.nchans, dtype=np.bool)
            kill_mask[kill_chans]= True

    else:
        logging.debug('No Kill Mask')
        kill_mask = None

    cand = Candidate(fil_name, snr=snr, width=width, dm=dm, label=label, tcand=tcand, kill_mask=kill_mask)
    cand.get_chunk()
    cand.fp.close()
    logging.info('Got Chunk')
    cand.dmtime()
    logging.info('Made DMT')
    if args.opt_dm:
        logging.info('Optimising DM')
        logging.warning('This feature is experimental!')
        cand.optimize_dm()
    else:
        cand.dm_opt = -1
        cand.snr_opt = -1
    cand.dedisperse()
    logging.info('Made Dedispersed profile')

    pulse_width = cand.width
    if pulse_width == 1:
        time_decimate_factor = 1
    else:
        time_decimate_factor = pulse_width // 2

    # Frequency - Time reshaping
    cand.decimate(key='ft', axis=0, pad=True, decimate_factor=time_decimate_factor, mode='median')
    crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - args.time_size // 2
    cand.dedispersed = crop(cand.dedispersed, crop_start_sample_ft, args.time_size, 0)
    logging.info(f'Decimated Time axis of FT to tsize: {cand.dedispersed.shape[0]}')

    if cand.dedispersed.shape[1] % args.frequency_size == 0:
        cand.decimate(key='ft', axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1] // args.frequency_size,
                      mode='median')
        logging.info(f'Decimated Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}')
    else:
        cand.resize(key='ft', size=args.frequency_size, axis=1, anti_aliasing=True)
        logging.info(f'Resized Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}')

    # DM-time reshaping
    cand.decimate(key='dmt', axis=1, pad=True, decimate_factor=time_decimate_factor, mode='median')
    crop_start_sample_dmt = cand.dmt.shape[1] // 2 - args.time_size // 2
    cand.dmt = crop(cand.dmt, crop_start_sample_dmt, args.time_size, 1)
    logging.info(f'Decimated DM-Time to dmsize: {cand.dmt.shape[0]} and tsize: {cand.dmt.shape[1]}')

    cand.dmt = normalise(cand.dmt)
    cand.dedispersed = normalise(cand.dedispersed)
    fout = label.strip('.fil')+'_dm_'+str(dm)+'_snr_'+str(snr)+'.h5'
    fout = cand.save_h5(out_dir=args.fout,fnout=fout)
    logging.info(fout)
    if args.plot:
        logging.info('Displaying the candidate')
        plot_h5(fout, show=False, save=True, detrend=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-fs', '--frequency_size', type=int, help='Frequency size after rebinning', default=256)
    parser.add_argument('-ts', '--time_size', type=int, help='Time length after rebinning', default=256)
    parser.add_argument('-c', '--cand_param_file', help='csv file with candidate parameters', type=str, required=True)
    parser.add_argument('-n', '--nproc', type=int, help='number of processors to use in parallel (default: 2)',
                        default=2)
    parser.add_argument('-p', '--plot', dest='plot', help='To display and save the candidates plots',
                        action='store_true')
    parser.add_argument('-o', '--fout', help='Output file directory for candidate h5', type=str)
    parser.add_argument('-opt', '--opt_dm', dest='opt_dm', help='Optimise DM', action='store_true', default=False)
    values = parser.parse_args()

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    cand_pars = pd.read_csv(values.cand_param_file,
                            names=['fil_file', 'snr', 'stime', 'dm', 'width', 'label', 'kill_mask_path'])
    process_list = []
    for index, row in cand_pars.iterrows():
        process_list.append(
            [row['fil_file'], row['snr'], 2 ** row['width'], row['dm'], row['label'], row['stime'],
             row['kill_mask_path'], values])
    with Pool(processes=values.nproc) as pool:
        pool.map(cand2h5, process_list, chunksize=1)
