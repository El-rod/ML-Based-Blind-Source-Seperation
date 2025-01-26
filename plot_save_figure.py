import os
import matplotlib.pyplot as plt
import numpy as np

all_sinr = np.arange(-30, 0.1, 3)
n_per_batch = 100
sig_len = 40960
soi_type = "QPSK"
interference_sig_type = "CommSignal2"
import pickle


def run_demod_test(sig1_est, bit1_est, soi_type='QPSK', interference_sig_type='CommSignal2',
                   testset_identifier='Seed0'):
    # For SampleEvalSet

    with open(f'dataset/Dataset_{testset_identifier}_{soi_type}_{interference_sig_type}.pkl', 'rb') as f:
        all_sig_mixture_groundtruth, all_sig1, all_bits1, _ = pickle.load(f)

    # Evaluation pipeline
    def eval_mse(all_sig_est, all_sig_soi):
        assert all_sig_est.shape == all_sig_soi.shape, 'Invalid SOI estimate shape'
        return np.mean(np.abs(all_sig_est - all_sig_soi) ** 2, axis=1)

    def eval_ber(bit_est, bit_true):
        ber = np.sum((bit_est != bit_true).astype(np.float32), axis=1) / bit_true.shape[1]
        assert bit_est.shape == bit_true.shape, 'Invalid bit estimate shape'
        return ber

    all_mse, all_ber = [], []
    for idx, sinr in enumerate(all_sinr):
        batch_mse = eval_mse(sig1_est[idx * n_per_batch:(idx + 1) * n_per_batch],
                             all_sig1[idx * n_per_batch:(idx + 1) * n_per_batch])
        bit_true_batch = all_bits1[idx * n_per_batch:(idx + 1) * n_per_batch]
        batch_ber = eval_ber(bit1_est[idx * n_per_batch:(idx + 1) * n_per_batch], bit_true_batch)
        all_mse.append(batch_mse)
        all_ber.append(batch_ber)

    all_mse, all_ber = np.array(all_mse), np.array(all_ber)
    mse_mean = 10 * np.log10(np.mean(all_mse, axis=-1))
    ber_mean = np.mean(all_ber, axis=-1)
    return mse_mean, ber_mean


for soi_type in ['QPSK']:
    for interference_sig_type in ['CommSignal2']:
        all_mse, all_ber = {}, {}
        for id_string in ['Default_TF_UNet', 'Default_Torch_WaveNet']:  #'Default_Torch_WaveNet' 'Default_TF_UNet'
            for testset_identifier in ['Seed0', 'Seed1', 'Seed2', 'Seed3', 'Seed4', 'Seed5']:
                sig1_est = np.load(os.path.join('outputs',
                                                f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}_{interference_sig_type}.npy'))
                bit1_est = np.load(os.path.join('outputs',
                                                f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}_{interference_sig_type}.npy'))
                mse_mean, ber_mean = run_demod_test(sig1_est, bit1_est, soi_type, interference_sig_type,
                                                    testset_identifier)
                if testset_identifier == 'Seed0':
                    all_mse[id_string] = mse_mean
                    all_ber[id_string] = ber_mean
                else:
                    all_mse[id_string] = np.vstack((mse_mean, all_mse[id_string]))
                    all_ber[id_string] = np.vstack((ber_mean, all_ber[id_string]))
            all_mse[id_string] = np.average(all_mse[id_string], axis=0)
            all_ber[id_string] = np.average(all_ber[id_string], axis=0)

        plt.figure()
        for id_string in all_mse.keys():
            plt.plot(all_sinr, all_mse[id_string], 'x--', label=id_string)
        plt.legend()
        plt.grid()
        plt.gca().set_ylim(top=3)
        plt.xlabel('SINR [dB]')
        plt.ylabel('MSE [dB]')
        plt.title(f'MSE - {soi_type} + {interference_sig_type}')
        plt.show()
        plt.savefig(os.path.join('outputs', 'mse.png'))

        plt.figure()
        for id_string in all_ber.keys():
            plt.semilogy(all_sinr, all_ber[id_string], 'x--', label=id_string)
        plt.legend()
        plt.grid()
        plt.ylim([1e-4, 1])
        plt.xlabel('SINR [dB]')
        plt.ylabel('BER')
        plt.title(f'BER - {soi_type} + {interference_sig_type}')
        plt.show()
        plt.savefig(os.path.join('outputs', 'ber.png'))
