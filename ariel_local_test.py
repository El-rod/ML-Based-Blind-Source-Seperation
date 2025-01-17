# generate SOI
# get interfernce
# save mixture

import pickle
import os
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

with open('dataset/Training_Dataset_QPSK_CommSignal3.pkl', 'rb') as f:
    all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = pickle.load(f)
    all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = all_sig_mixture[-2:-1,
                                                                              :], all_sig1_groundtruth[-2:-1,
                                                                                  :], all_bits1_groundtruth[-2:-1,
                                                                                      :], meta_data[-2:-1, :]

# load Unet model for mixture:
#all_sinr = np.arange(-30, 0.1, 3)
all_sinr = [3]
n_per_batch = 100
sig_len = 40960
soi_type = "QPSK"
interference_sig_type = "CommSignal3"
from rfcutils.qpsk_helper_fn import qpsk_matched_filter_demod


def run_demod_test(sig1_est, bit1_est):
    # For SampleEvalSet

    with open('dataset/Training_Dataset_QPSK_CommSignal3.pkl', 'rb') as f:
        all_sig_mixture_groundtruth, all_sig1, all_bits1, _ = pickle.load(f)
        all_sig_mixture_groundtruth, all_sig1, all_bits1 = all_sig_mixture_groundtruth[0:1, :], all_sig1[0:1,
                                                                                                :], all_bits1[0:1, :]

    # Evaluation pipeline
    def eval_mse(all_sig_est, all_sig_soi):
        assert all_sig_est.shape == all_sig_soi.shape, 'Invalid SOI estimate shape'
        return np.mean(np.abs(all_sig_est - all_sig_soi) ** 2, axis=1)

    def eval_ber(bit_est, bit_true):
        ber = np.sum((bit_est != bit_true).astype(np.float32), axis=1) / bit_true.shape[1]
        assert bit_est.shape == bit_true.shape, 'Invalid bit estimate shape'
        return ber

    all_mse, all_ber = [], []
    idx = 0
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


# do the magic, save as output
def run_inference_unet(all_sig_mixture):
    from src import unet_model as unet

    nn_model = unet.get_unet_model((sig_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003)
    nn_model.load_weights(
        os.path.join('models', f'dataset_{soi_type.lower()}_{interference_sig_type.lower()}_mixture_unet',
                     'checkpoint')).expect_partial()

    all_sig_mixture = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)
    sig1_out = nn_model.predict(all_sig_mixture, batch_size=100, verbose=1)
    sig1_est = tf.complex(sig1_out[:, :, 0], sig1_out[:, :, 1])

    bit_est = []
    for idx, sinr_db in tqdm(enumerate(all_sinr)):
        bit_est_batch, _ = qpsk_matched_filter_demod(sig1_est[idx * n_per_batch:(idx + 1) * n_per_batch])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est


def run_inference_wavenet(all_sig_mixture):
    from omegaconf import OmegaConf
    from src.config_torchwavenet import Config, parse_configs
    from src.torchwavenet import Wave

    cfg = OmegaConf.load("src/configs/wavenet.yml")
    cli_cfg = None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))
    cfg.model_dir = f"torchmodels/dataset_{soi_type.lower()}_{interference_sig_type.lower()}_mixture_wavenet"
    nn_model = Wave(cfg.model).cpu()
    nn_model.load_state_dict(torch.load(cfg.model_dir + "/weights.pt", map_location=torch.device('cpu'))['model'])

    sig_mixture_comp = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)
    with torch.no_grad():
        nn_model.eval()
        all_sig1_out = []
        bsz = 100
        for i in tqdm(range(sig_mixture_comp.shape[1] // bsz)): # was 0 but when i changed it sig_len is now at 1
            sig_input = torch.Tensor(sig_mixture_comp[i * bsz:(i + 1) * bsz].numpy()).transpose(1, 2).to('cpu')
            sig1_out = nn_model(sig_input)
            all_sig1_out.append(sig1_out.transpose(1, 2).detach().cpu().numpy())
    sig1_out = tf.concat(all_sig1_out, axis=0)
    sig1_est = tf.complex(sig1_out[:, :, 0], sig1_out[:, :, 1])

    bit_est = []
    for idx, sinr_db in tqdm(enumerate(all_sinr)):
        bit_est_batch, _ = qpsk_matched_filter_demod(sig1_est[idx * n_per_batch:(idx + 1) * n_per_batch])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est

sig1_est, bit_est = run_inference_unet(all_sig_mixture)

# eval with ground truth
print(1)
mse_mean, ber_mean = run_demod_test(sig1_est, bit_est)
print(mse_mean, ber_mean)
