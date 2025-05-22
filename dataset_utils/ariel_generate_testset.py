"""
A python script for generating example mixtures for training;
this script creates a training set that is more aligned with the TestSet’s specifications
(e.g., focusing solely on the 11 discrete target SINR levels).
This script saves a pickle file dataset/Training_Dataset_[SOI Type]_[Interference Type].pkl'
that contains all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data.
"""

import os, sys
import numpy as np
import random
import h5py
from tqdm import tqdm
import pickle
import argparse
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rfcutils

# math functions
get_db = lambda p: 10 * np.log10(p)
get_pow = lambda s: np.mean(np.abs(s) ** 2, axis=-1)
get_sinr = lambda s, i: get_pow(s) / get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s, i))

sig_len = 40960
default_n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)


def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len // 16)
        demod_soi = rfcutils.qpsk_matched_filter_demod
    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len // 16)
        demod_soi = rfcutils.qam16_matched_filter_demod
    elif soi_sig_type == 'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len // 4)
        demod_soi = rfcutils.qpsk2_matched_filter_demod
    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len // 80)
        _, _, _, RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len // 80)
        demod_soi = lambda s: rfcutils.ofdm_demod(s, RES_GRID)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi, demod_soi


def generate_demod_testmixture(soi_type, interference_sig_type, n_per_batch=default_n_per_batch, seed_number=0):
    generate_soi, demod_soi = get_soi_generation_fn(soi_type)
    with h5py.File(os.path.join('dataset', 'interferenceset_frame', interference_sig_type + '_raw_data.h5'),
                   'r') as data_h5file:
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")

    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.random.set_seed(seed_number)

    all_sig_mixture, all_sig1, all_bits1, meta_data = [], [], [], []
    for idx, sinr in tqdm(enumerate(all_sinr)):
        sig1, _, bits1, _ = generate_soi(n_per_batch, sig_len)
        sig2 = sig_data[np.random.randint(sig_data.shape[0], size=(n_per_batch)), :]

        sig_target = sig1[:, :sig_len]

        rand_start_idx2 = np.random.randint(sig2.shape[1] - sig_len, size=sig2.shape[0])
        inds2 = tf.cast(rand_start_idx2.reshape(-1, 1) + np.arange(sig_len).reshape(1, -1), tf.int32)
        sig_interference = tf.experimental.numpy.take_along_axis(sig2, inds2, axis=1)

        # Interference Coefficient
        rand_gain = np.sqrt(10 ** (-sinr / 10)).astype(np.float32)
        rand_phase = tf.random.uniform(shape=[sig_interference.shape[0], 1])
        rand_gain = tf.complex(rand_gain, tf.zeros_like(rand_gain))
        rand_phase = tf.complex(rand_phase, tf.zeros_like(rand_phase))
        coeff = rand_gain * tf.math.exp(1j * 2 * np.pi * rand_phase)

        sig_mixture = sig_target + sig_interference * coeff

        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig_target)
        all_bits1.append(bits1)

        actual_sinr = get_sinr_db(sig_target, sig_interference * coeff)
        meta_data.append(np.vstack(([rand_gain.numpy().real for _ in range(n_per_batch)],
                                    [sinr for _ in range(n_per_batch)], actual_sinr,
                                    [soi_type for _ in range(n_per_batch)],
                                    [interference_sig_type for _ in range(n_per_batch)])))

    return all_sig_mixture, all_sig1, all_bits1, meta_data


def generate_combined_testmixture(soi_type, interference_type1, interference_type2, p, n_per_batch=default_n_per_batch,
                                  seed=0):
    all_sig_mixture, all_sig, all_bits, meta_data = generate_demod_testmixture(soi_type, interference_type1,
                                                                               round(n_per_batch * p), seed)
    all_sig_mixture2, all_sig2, all_bits2, meta_data2 = generate_demod_testmixture(soi_type, interference_type2,
                                                                                   round(n_per_batch * (1 - p)), seed)
    all_sig_mixture.extend(all_sig_mixture2)
    all_sig.extend(all_sig2)
    all_bits.extend(all_bits2)
    meta_data.extend(meta_data2)

    with tf.device('CPU'):
        all_sig_mixture = tf.concat(all_sig_mixture, axis=0).numpy()
        all_sig = tf.concat(all_sig, axis=0).numpy()
        all_bits = tf.concat(all_bits, axis=0).numpy()

    meta_data = np.concatenate(meta_data, axis=1).T

    # sort by snr
    sort_indices = meta_data[:, 1].astype('float64').argsort()
    all_sig_mixture = all_sig_mixture[sort_indices, :]
    all_sig = all_sig[sort_indices, :]
    all_bits = all_bits[sort_indices, :]
    meta_data = meta_data[sort_indices, :]

    pickle.dump((all_sig_mixture, all_sig, all_bits, meta_data),
                open(os.path.join('dataset',
                                  f'Dataset_Seed{seed}_{soi_type}_{interference_type1}+{interference_type2}.pkl'),
                     'wb'),
                protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-b', '--n_per_batch', default=100, type=int, help='')
    parser.add_argument('-seed', '--random_seed', default=0, type=int, help='')
    parser.add_argument('-soi', '--soi_sig_type', help='')
    parser.add_argument('-i_sig1', '--interference_sig_type1', help='')
    parser.add_argument('-i_sig2', '--interference_sig_type2', help='')
    parser.add_argument('-p', '--mixtures_probability', default=0.5, type=float, help='')

    args = parser.parse_args()

    # generate_demod_testmixture(args.soi_sig_type, args.interference_sig_type, args.n_per_batch, args.random_seed)
    # generate_combined_testmixture(args.soi_sig_type, args.interference_sig_type1, args.interference_sig_type2, args.mixtures_probability, args.n_per_batch, args.random_seed)
    generate_combined_testmixture('QPSK', 'CommSignal2', 'EMISignal1', 0.5, 1000, 5000)
