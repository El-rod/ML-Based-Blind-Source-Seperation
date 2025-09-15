import os
import sys
import tensorflow_datasets as tfds
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)

#from src import unet_model as unet
from src import unet_8layered_model as unet

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"])

bsz = 32
EPOCHS = 18
suffix = 'unet8_MT'
all_datasets = ['dataset_qpsk_commsignal3_emisignal1_mixture']


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

    ds_train, _ = tfds.load(dataset_type, split="train[:90%]",
                            shuffle_files=True,
                            as_supervised=False,  # True
                            with_info=True,
                            data_dir='serverdata/tfds'
                            )
    ds_val, _ = tfds.load(dataset_type, split="train[90%:]",
                          shuffle_files=True,
                          as_supervised=False,  # True
                          with_info=True,
                          data_dir='serverdata/tfds'
                          )

    # def extract_example(mixture, target):
    #     return mixture, target

    def extract_example(example):
        return example['mixture'], example['signal']

    ds_train = ds_train.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(bsz)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(bsz)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    window_len = 40960
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    model_pathname = os.path.join('models', f'{dataset_type}_{suffix}', 'checkpoint')
    checkpoint = ModelCheckpoint(filepath=model_pathname, monitor='val_loss', verbose=0, save_best_only=True,
                                 mode='min', save_weights_only=True)

    import datetime
    log_pathname = os.path.join('models', f'{dataset_type}_{suffix}',
                                f'log{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    loss_logger = LossSummaryCallback(log_dir=log_pathname)

    with mirrored_strategy.scope():
        nn_model = unet.get_unet_model((window_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003)

        if os.path.exists(model_pathname):
            nn_model.load_weights(model_pathname).expect_partial()

        nn_model.fit(ds_train, epochs=EPOCHS, batch_size=bsz, shuffle=True, verbose=1, validation_data=ds_val,
                     callbacks=[checkpoint, earlystopping, loss_logger])


if __name__ == '__main__':
    train_script(int(0))
