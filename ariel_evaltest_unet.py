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

from rfcutils.qpsk_helper_fn import qpsk_matched_filter_demod


# do the magic, save as output
def run_inference_unet(all_sig_mixture, model, n_per_batch=100):
    with tf.device('/gpu:1'):
        from src import unet_model as unet
        # from src import ariel_unet_model as unet

        nn_model = unet.get_unet_model((sig_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003)
        model_path = os.path.join('models', model, 'checkpoint')
        nn_model.load_weights(model_path).expect_partial()

        all_sig_mixture = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)

        # sig1_out = nn_model.predict(all_sig_mixture, batch_size=n_per_batch)

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
        for idx, sinr_db in tqdm(enumerate(all_sinr)):
            bit_est_batch, _ = qpsk_matched_filter_demod(sig1_est[idx * n_total:(idx + 1) * n_total])
            bit_est.append(bit_est_batch)
        bit_est = tf.concat(bit_est, axis=0)
        sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
        return sig1_est, bit_est


if __name__ == "__main__":
    # soi_type, interference_sig_type = sys.argv[1], sys.argv[2]

    # testset_identifier = 'Seed20250401'
    testset_identifier = 'Seed5000'
    soi_type = "QPSK"
    interference_sig_type1 = "CommSignal2"
    interference_sig_type2 = "EMISignal1"
    classify = 0
    n_per_batch = 100

    with open(f'dataset/Dataset_{testset_identifier}_{soi_type}_{interference_sig_type1}+{interference_sig_type2}.pkl',
              'rb') as f:
        all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = pickle.load(f)

    if classify:
        # interference_types = np.unique(meta_data[:,4])
        interference_types = np.array(["CommSignal2", "EMISignal1"])

        model1 = f'dataset_{soi_type.lower()}_{interference_types[0].lower()}_mixture_unet'
        model2 = f'dataset_{soi_type.lower()}_{interference_types[1].lower()}_mixture_unet'

        from ariel_evaltest_classifier import run_inference_classifier as classifier_predict

        predicted_sig_type_index = classifier_predict(all_sig_mixture, 2)
        print(predicted_sig_type_index.shape)
        predicted_sig_type_labels = interference_types[predicted_sig_type_index]

        sig_type1_indicies = (predicted_sig_type_labels == interference_types[0]).nonzero()[0]
        sig_type2_indicies = (predicted_sig_type_labels == interference_types[1]).nonzero()[0]

        # sig_type1_indicies = (meta_data[:,4]==interference_types[0]).nonzero()[0]
        # sig_type2_indicies = (meta_data[:,4]==interference_types[1]).nonzero()[0]

        all_sig_mixture_type1 = all_sig_mixture[sig_type1_indicies]
        all_sig_mixture_type2 = all_sig_mixture[sig_type2_indicies]
        p = sig_type1_indicies.shape[0] / all_sig_mixture.shape[0]

        sig1_est, bit1_est = run_inference_unet(all_sig_mixture_type1, model1, round(n_per_batch * p))
        sig2_est, bit2_est = run_inference_unet(all_sig_mixture_type2, model2, round(n_per_batch * (1 - p)))

        sig_est = np.empty((all_sig_mixture.shape[0], sig1_est.shape[1]), dtype=np.complex128)
        bit_est = np.empty((all_sig_mixture.shape[0], bit1_est.shape[1]), dtype=np.complex128)

        sig_est[sig_type1_indicies] = sig1_est
        sig_est[sig_type2_indicies] = sig2_est

        bit_est[sig_type1_indicies] = bit1_est
        bit_est[sig_type2_indicies] = bit2_est

        id_string = 'Default_TF_UNet'
    else:
        # model = f'dataset_{soi_type.lower()}_{interference_sig_type1.lower()}+{interference_sig_type2.lower()}_mixture_unet_ariel'
        model = 'dataset_qpsk_comm2and_emi1_mixture_unet_ariel'
        sig_est, bit_est = run_inference_unet(all_sig_mixture, model)
        id_string = 'TwoMixTrained_TF_UNet'

    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}_{interference_sig_type1}+{interference_sig_type2}'),
            sig_est)
    np.save(os.path.join('outputs',
                         f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}_{interference_sig_type1}+{interference_sig_type2}'),
            bit_est)
