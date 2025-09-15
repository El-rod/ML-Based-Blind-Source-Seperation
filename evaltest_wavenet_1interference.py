import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

DEVICE = 'cuda:1'
all_sinr = np.arange(-30, 0.1, 3)
sig_len = 40960
n_per_batch = 100


def get_soi_demod_fn(soi_sig_type):
    import rfcutils
    if soi_sig_type == 'QPSK':
        from rfcutils import qpsk_matched_filter_demod
        demod_soi = qpsk_matched_filter_demod
    elif soi_sig_type == 'QAM16':
        from rfcutils import qam16_matched_filter_demod
        demod_soi = qam16_matched_filter_demod
    elif soi_sig_type == 'QPSK2':
        from rfcutils import qpsk2_matched_filter_demod
        demod_soi = qpsk2_matched_filter_demod
    elif soi_sig_type == '8PSK':
        from rfcutils import _8psk_matched_filter_demod
        demod_soi = _8psk_matched_filter_demod
    elif soi_sig_type == '16PSK':
        from rfcutils import _16psk_matched_filter_demod
        demod_soi = _16psk_matched_filter_demod
    else:
        raise Exception("SOI Type not recognized")
    return demod_soi


def run_inference_wavenet(all_sig_mixture, model_path):
    from omegaconf import OmegaConf
    from src.config_torchwavenet import Config, parse_configs
    from src.torchwavenet import Wave

    cfg = OmegaConf.load("src/configs/wavenet.yml")
    cli_cfg = None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))
    cfg.model_dir = model_path
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
    demod_soi = get_soi_demod_fn(soi_type)
    for idx, sinr_db in tqdm(enumerate(all_sinr)):
        bit_est_batch, _ = demod_soi(sig1_est[idx * n_total:(idx + 1) * n_total])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est


if __name__ == "__main__":

    testset_identifier = 'Seed1'
    soi_type = "QPSK"
    interference_sig_type = "CommSignal2"
    type2 = "EMISignal1"

    classify = 0

    with open(f'dataset/Dataset_{testset_identifier}_{soi_type}+{interference_sig_type}.pkl', 'rb') as f:
        all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = pickle.load(f)

    if classify:
        model_path = f"torchmodels/dataset_{soi_type.lower()}_{interference_sig_type.lower()}_mixture_wavenet"
        id_string = 'ConditionedTrained_WaveNet'
    else:
        model_path = f"torchmodels/dataset_{soi_type.lower()}_{interference_sig_type.lower()}_{type2.lower()}_mixture_wavenet_MT"

        if not os.path.exists(model_path):
            model_path = f'torchmodels/dataset_{soi_type.lower()}_{type2.lower()}_{interference_sig_type.lower()}_mixture_wavenet_MT'

        id_string = f'MixtureTrained_with_{type2}_WaveNet'

    sig1_est, bit1_est = run_inference_wavenet(all_sig_mixture, model_path)

    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}+{interference_sig_type}'),
            sig1_est)
    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}+{interference_sig_type}'),
            bit1_est)
