import pickle
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

n_per_batch = 100
sig_len = 40960
all_sinr = np.arange(-30, 0.1, 3)


def run_inference_classifier(all_sig_mixture, NUM_CLASSES=2):
    with tf.device('/gpu:0'):
        from src import cnn_detector

        nn_model = ariel_cnn_classifier.get_classifier_model((sig_len, 2), NUM_CLASSES)
        nn_model.load_weights(
            os.path.join('models', f'dataset_qpsk_comm2and_emi1_mixture_cnn_classifier_ariel',
                         'checkpoint')).expect_partial()

        all_sig_mixture = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)

        n_total = all_sig_mixture.shape[0]
        outputs = []
        for i in tqdm(range(0, n_total, n_per_batch)):
            batch = all_sig_mixture[i:i + n_per_batch]
            batch_out = nn_model.predict(batch, batch_size=batch.shape[0], verbose=0)
            outputs.append(batch_out)
        class_logits = np.concatenate(outputs, axis=0)

        hard_decision = tf.argmax(class_logits, axis=1).numpy().astype(int)

        return hard_decision


if __name__ == "__main__":

    testset_identifier = 'Seed5000'
    soi_type = "QPSK"
    interference_sig_type1 = "CommSignal2"
    interference_sig_type2 = "EMISignal1"
    NUM_CLASSES = 2

    with open(f'dataset/Dataset_{testset_identifier}_{soi_type}_{interference_sig_type1}+{interference_sig_type2}.pkl',
              'rb') as f:
        all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data = pickle.load(f)

    predicted_classes = run_inference_classifier(all_sig_mixture, NUM_CLASSES)
    id_string = 'a'


    def eval_error(pred_labels, true_labels):
        assert pred_labels.shape == true_labels.shape, 'Mismatched label shapes'
        error_rate = np.mean(pred_labels != true_labels)
        return error_rate


    true_label = (meta_data[:, 4] == "EMISignal1").astype(int)
    all_error = []
    bsz = 2000
    for idx, sinr in enumerate(all_sinr):
        batch_error = eval_error(predicted_classes[idx * bsz:(idx + 1) * bsz],
                                 true_label[idx * bsz:(idx + 1) * bsz])
        all_error.append(batch_error)

    plt.figure()
    plt.plot(all_sinr, all_error, 'x--', label=f'{id_string}')
    plt.legend()
    plt.grid()
    plt.gca().set_ylim(top=3)
    plt.xlabel('SINR [dB]')
    plt.ylabel('P_k ')
    plt.title(f'error class')
    plt.show()
    plt.savefig(os.path.join('outputs', f'class_error.png'))

