import pickle
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

all_sinr = np.arange(-30, 0.1, 3)
sig_len = 40960
n_per_batch = 100


# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[1], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[1], True)


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


def run_inference_unet(all_sig_mixture, model):
    with tf.device('/gpu:0'):
        if extra_layers:
            from src import unet_8layered_model as unet
        else:
            from src import unet_model as unet

        nn_model = unet.get_unet_model((sig_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003)
        model_path = os.path.join('models', model, 'checkpoint')
        nn_model.load_weights(model_path).expect_partial()

        all_sig_mixture = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)

        # split prediction into smaller batches
        n_total = all_sig_mixture.shape[0]
        outputs = []
        for i in tqdm(range(0, n_total, n_per_batch)):
            batch = all_sig_mixture[i:i + n_per_batch]
            batch_out = nn_model.predict(batch, batch_size=batch.shape[0], verbose=0)
            outputs.append(batch_out)
        sig1_out = np.concatenate(outputs, axis=0)

        sig1_est = tf.complex(sig1_out[:, :, 0], sig1_out[:, :, 1])

        bit_est = []
        demod_soi = get_soi_demod_fn(soi_type)
        for idx, sinr_db in tqdm(enumerate(all_sinr)):
            bit_est_batch, _ = demod_soi(
                sig1_est[idx * (n_total // len(all_sinr)):(idx + 1) * (n_total // len(all_sinr))])
            bit_est.append(bit_est_batch)
        bit_est = tf.concat(bit_est, axis=0)
        sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
        return sig1_est, bit_est


if __name__ == "__main__":

    soi_type = "QPSK"
    interference_sig_type = 'CommSignal3'  # "CommSignal2" "EMISignal1" "CommSignal3" "CommSignal5G1"
    testset_identifier = 'Seed1'
    classify = 0
    type2 = 'EMISignal1'  # "CommSignal2" "EMISignal1" "CommSignal3" "CommSignal5G1"
    extra_layers = 0

    with open(f'dataset/Dataset_{testset_identifier}_{soi_type}+{interference_sig_type}.pkl', 'rb') as f:
        all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = pickle.load(f)

    if classify:
        model = f'dataset_{soi_type.lower()}_{interference_sig_type.lower()}_mixture_unet'
        id_string = 'ConditionedTrained_UNet'
    else:
        if extra_layers:
            suffix = '8_MT'
            id_string = f'MixtureTrained_with_{type2}_8Layered_UNet'
        else:
            suffix = '_MT'
            id_string = f'MixtureTrained_with_{type2}_UNet'

        model = f'dataset_{soi_type.lower()}_{interference_sig_type.lower()}_{type2.lower()}_mixture_unet{suffix}'

        if not os.path.exists(os.path.join('models', model, 'checkpoint')):
            model = f'dataset_{soi_type.lower()}_{type2.lower()}_{interference_sig_type.lower()}_mixture_unet{suffix}'

    sig1_est, bit1_est = run_inference_unet(all_sig_mixture, model)

    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}+{interference_sig_type}'),
            sig1_est)
    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}+{interference_sig_type}'),
            bit1_est)
