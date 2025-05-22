import tensorflow as tf
from tensorflow.keras import layers, models

def get_classifier_model(input_shape, num_classes, k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003):
    in0 = layers.Input(shape=input_shape)
    x = in0

    x = layers.BatchNormalization()(x)

    for n_layer, k in enumerate([8, 8, 8, 8, 8]):
        if n_layer == 0:
            conv = layers.Conv1D(k_neurons * k, long_k_sz, activation="relu", padding="same")(x)
        else:
            conv = layers.Conv1D(k_neurons * k, k_sz, activation="relu", padding="same")(x)

        conv = layers.Conv1D(k_neurons * k, k_sz, activation="relu", padding="same")(conv)
        pool = layers.MaxPooling1D(2)(conv)

        if n_layer == 0:
            pool = layers.Dropout(0.25)(pool)
        else:
            pool = layers.Dropout(0.5)(pool)

        x = pool

    # bottleneck
    convm = layers.Conv1D(k_neurons * 8, k_sz, activation="relu", padding="same")(x)
    convm = layers.Conv1D(k_neurons * 8, k_sz, activation="relu", padding="same")(convm)

    x = convm

    # classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=in0, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model
