import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

all_sinr = np.arange(-30, 0.1, 3)
sig_len = 40960

from rfcutils.qpsk_helper_fn import qpsk_matched_filter_demod

DEVICE = 'cuda:0'


def run_inference_wavenet(all_sig_mixture, model, n_per_batch=100):
    from omegaconf import OmegaConf
    from src.config_torchwavenet import Config, parse_configs
    from src.torchwavenet import Wave

    cfg = OmegaConf.load("src/configs/wavenet.yml")
    cli_cfg = None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))
    cfg.model_dir = f"torchmodels/{model}"
    nn_model = Wave(cfg.model).to(DEVICE)
    nn_model.load_state_dict(torch.load(cfg.model_dir + "/weights.pt")['model'])

    sig_mixture_comp = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)
    n_total = sig_mixture_comp.shape[0]
    with torch.no_grad():
        nn_model.eval()
        all_sig1_out = []
        for i in tqdm(range(0, n_total, n_per_batch)):
            sig_input = torch.Tensor(sig_mixture_comp[i:i + n_per_batch].numpy()).transpose(1, 2).to(DEVICE)
            sig1_out = nn_model(sig_input)
            all_sig1_out.append(sig1_out.transpose(1, 2).detach().cpu().numpy())
    sig1_out = np.concatenate(all_sig1_out, axis=0)
    sig1_est = tf.complex(sig1_out[:, :, 0], sig1_out[:, :, 1])

    bit_est = []
    for idx, sinr_db in tqdm(enumerate(all_sinr)):
        bit_est_batch, _ = qpsk_matched_filter_demod(sig1_est[idx * n_total:(idx + 1) * n_total])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est


if __name__ == "__main__":
    # soi_type, interference_sig_type = sys.argv[1], sys.argv[2]

    testset_identifier = 'Seed20250401'
    # testset_identifier = 'Seed0'
    soi_type = "QPSK"
    interference_sig_type1 = "CommSignal2"
    interference_sig_type2 = "EMISignal1"
    classify = 1
    n_per_batch = 100

    with open(f'dataset/Dataset_{testset_identifier}_{soi_type}_{interference_sig_type1}+{interference_sig_type2}.pkl',
              'rb') as f:
        all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = pickle.load(f)

    if classify:
        interference_types = np.unique(meta_data[:, 4])
        model1 = f'dataset_{soi_type.lower()}_{interference_types[0].lower()}_mixture_wavenet'
        model2 = f'dataset_{soi_type.lower()}_{interference_types[1].lower()}_mixture_wavenet'
        sig_type1_indicies = (meta_data[:, 4] == interference_types[0]).nonzero()[0]
        sig_type2_indicies = (meta_data[:, 4] == interference_types[1]).nonzero()[0]

        all_sig_mixture_type1 = all_sig_mixture[sig_type1_indicies]
        all_sig_mixture_type2 = all_sig_mixture[sig_type2_indicies]
        p = sig_type1_indicies.shape[0] / all_sig_mixture.shape[0]

        sig1_est, bit1_est = run_inference_wavenet(all_sig_mixture_type1, model1, round(n_per_batch * p))
        sig2_est, bit2_est = run_inference_wavenet(all_sig_mixture_type2, model2, round(n_per_batch * (1 - p)))

        sig_est = np.empty((all_sig_mixture.shape[0], sig1_est.shape[1]), dtype=np.complex128)
        bit_est = np.empty((all_sig_mixture.shape[0], bit1_est.shape[1]), dtype=np.complex128)

        sig_est[sig_type1_indicies] = sig1_est
        sig_est[sig_type2_indicies] = sig2_est

        bit_est[sig_type1_indicies] = bit1_est
        bit_est[sig_type2_indicies] = bit2_est

        id_string = 'Default_Torch_WaveNet'
    else:
        # model = f'dataset_{soi_type.lower()}_{interference_sig_type1.lower()}+{interference_sig_type2.lower()}_mixture_wavenet_ariel'
        model = 'dataset_qpsk_comm2andemi1_mixture_wavenet_ariel'
        sig_est, bit_est = run_inference_wavenet(all_sig_mixture, model)
        id_string = 'TwoMixTrained_Torch_WaveNet'

    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}_{interference_sig_type1}+{interference_sig_type2}'),
            sig_est)
    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}_{interference_sig_type1}+{interference_sig_type2}'),
            bit_est)
