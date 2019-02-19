#!/usr/bin/env python3

import argparse
import pandas as pd
from multiprocessing import Pool
from candidate import Candidate, crop
import logging
from h5plotter import plot_h5
logger = logging.getLogger()
logger = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s')

def cand2h5(cand_val, args):
    """
    TODO: Add option to use cand.resize for reshaping FT and DMT
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :type args: object
    :param cand_val: List of candidate parameters (fil_name, snr, width, dm, label, tcand(s))
    :param args: Input arguments for the candidate
    :return:
    """
    fil_name, snr, width, dm, label, tcand = cand_val
    cand = Candidate(fil_name, snr=snr, width=width, dm=dm, label=label, tcand=tcand)
    cand.get_chunk()
    cand.fp.close()
    logging.info('Got Chunk')
    cand.dmtime()
    logging.info('Made DMT')
    if args.opt_dm:
        logging.info('Optimising DM')
        cand.optimize_dm()
    else:
        cand.dm_opt = -1
        cand.snr_opt = -1
    cand.dedisperse()
    logging.info('Made Dedispersed profile')

    if args.resize:
        cand.resize(key='ft', size=args.time_size, axis=0, anti_aliasing=True)
        cand.resize(key='ft', size=args.frequency_size, axis=1, anti_aliasing=True)
        logging.info(f'Resized Frequency-Time data to fsize: {args.frequency_size} and tsize: {args.time_size}')
        cand.resize(key='dmt', size = args.time_size, axis=0, anti_aliasing=True)
        logging.info(f'Resized DM-Time to dmsize: 256 and tsize: {args.time_size}')
    else:
        pulse_width = cand.width
        if pulse_width == 1:
            time_decimate_factor = 1
        else:
            time_decimate_factor = pulse_width // 2
        cand.decimate(key='ft', axis=0, pad=True, decimate_factor=time_decimate_factor, mode='median')
        crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - args.time_size // 2
        cand.dedispersed = crop(cand.dedispersed, crop_start_sample_ft, args.time_size, 0)
        cand.decimate(key='ft', axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1]//args.frequency_size, mode='median')
        logging.info(f'Decimated Frequency-Time data to fsize: {args.frequency_size} and tsize: {args.time_size}')

        cand.decimate(key='dmt', axis=1, pad=True, decimate_factor=time_decimate_factor)
        crop_start_sample_dmt = cand.dmt.shape[1] // 2 - args.time_size // 2
        cand.dmt = crop(cand.dmt, crop_start_sample_dmt, args.time_size, 1)
        logging.info(f'Decimated DM-Time to dmsize: 256 and tsize: {args.time_size}')

    fout = cand.save_h5(out_dir=args.fout)
    logging.info(fout)
    if args.plot:
        logging.info('Displaying the candidate')
        plot_h5(fout, show=True, save=True, detrend=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', '--frequency_size', type=int, help='Frequency size after rebinning', default=256)
    parser.add_argument('-ts', '--time_size', type=int, help='Time length after rebinning', default=256)
    parser.add_argument('-c', '--cand_param_file', help='csv file with candidate parameters', type=str, required=True)
    parser.add_argument('-n', '--nproc', type=int, help='number of processors to use in parallel (default: 2)',
                        default=2)
    parser.add_argument('-p', '--plot', dest='plot', help='To display and save the candidates plots',
                        action='store_true')
    parser.add_argument('-o', '--fout', help='Output file directory for candidate h5', type=str)
    parser.add_argument('-opt', '--opt_dm', dest='opt_dm', help='Optimise DM', action='store_true', default=False)
    parser.add_argument('-r', '--resize', dest='resize', help='Reshape Frequency-Time and DM-Time using skimage.'
                                                              'transform resize', action='store_true', default=False)
    values = parser.parse_args()

    cand_pars = pd.read_csv(values.cand_param_file, names=['fil_file', 'snr', 'stime', 'dm', 'width', 'label'])
    process_list = []
    arg_list = []
    for index, row in cand_pars.iterrows():
        process_list.append([row['fil_file'], row['snr'], 2**row['width'], row['dm'], row['label'], row['stime']])
        arg_list.append(values)
    with Pool(processes=values.nproc) as pool:
        pool.starmap(cand2h5, zip(process_list, arg_list), chunksize=1)
