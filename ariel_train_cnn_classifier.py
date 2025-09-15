import os
import sys
import tensorflow_datasets as tfds
import tensorflow as tf

from src import cnn_detector as cnn

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"])

bsz = 32  # 32, 64
EPOCHS = 6

# all_datasets = ['QPSK_CommSignal2', 'QPSK2_CommSignal2', 'QAM16_CommSignal2', 'OFDMQPSK_CommSignal2',
#                 'QPSK_CommSignal3', 'QPSK2_CommSignal3', 'QAM16_CommSignal3', 'OFDMQPSK_CommSignal3',
#                 'CommSignal2_CommSignal3',
#                 'QPSK_EMISignal1', 'QPSK2_EMISignal1', 'QAM16_EMISignal1', 'OFDMQPSK_EMISignal1',
#                 'CommSignal2_EMISignal1',
#                 'QPSK_CommSignal5G1', 'QPSK2_CommSignal5G1', 'QAM16_CommSignal5G1', 'OFDMQPSK_CommSignal5G1',
#                 'CommSignal2_CommSignal5G1']

all_datasets = ['dataset_qpsk_comm2and_emi1_mixture']


class LossSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            if 'loss' in logs:
                tf.summary.scalar('train/loss', logs['loss'], step=epoch)
            if 'val_loss' in logs:
                tf.summary.scalar('val/loss', logs['val_loss'], step=epoch)
        self.writer.flush()


def train_script(idx):
    dataset_type = all_datasets[idx]

    ds_train, ds_info = tfds.load(dataset_type, split="train[:90%]",
                                  shuffle_files=True,
                                  as_supervised=False,
                                  with_info=True,
                                  data_dir='tfds'
                                  )
    ds_val, _ = tfds.load(dataset_type, split="train[90%:]",
                          shuffle_files=True,
                          as_supervised=False,
                          with_info=True,
                          data_dir='tfds'
                          )

    def extract_example(example):
        return example['mixture'], example['sig_type']

    SIGNAL_TYPE_NUM = ds_info.features['sig_type'].num_classes

    ds_train = ds_train.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(bsz)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(bsz)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    window_len = 40960
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    model_pathname = os.path.join('models', f'{dataset_type}_cnn_classifier_ariel', 'checkpoint')
    checkpoint = ModelCheckpoint(filepath=model_pathname, monitor='val_loss', verbose=0, save_best_only=True,
                                 mode='min', save_weights_only=True)

    import datetime
    log_pathname = os.path.join('models', f'{dataset_type}_unet_ariel',
                                f'log{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    loss_logger = LossSummaryCallback(log_dir=log_pathname)

    with mirrored_strategy.scope():
        nn_model = cnn.get_classifier_model((window_len, 2), SIGNAL_TYPE_NUM, k_sz=3, long_k_sz=101, k_neurons=32,
                                            lr=0.0003)

        if os.path.exists(model_pathname):
            nn_model.load_weights(model_pathname).expect_partial()

        nn_model.fit(ds_train, epochs=EPOCHS, batch_size=bsz, shuffle=True, verbose=1, validation_data=ds_val,
                     callbacks=[checkpoint, earlystopping, loss_logger])


if __name__ == '__main__':
    # train_script(int(sys.argv[1]))
    train_script(int(0))
