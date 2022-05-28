import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Flatten, Dense, Reshape, \
    Conv2DTranspose, InputLayer, MaxPooling2D, GlobalMaxPooling2D, Dropout, Lambda
from tensorflow.keras import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import os
import tensorflow.keras.backend as K


def run_models_main(parameters, data):
    models = {}
    metrics = {}
    # convolutional autoencoder
    encoded, reconstruction_metrics, convolutional_autoencoder = run_convolutional_autoencoder(parameters, data)
    models['convolutional_autoencoder'] = convolutional_autoencoder
    models['encoder'] = convolutional_autoencoder.encoder
    metrics['reconstruction_metrics'] = reconstruction_metrics
    # classifier
    downstream_classifier, classification_metrics = run_classifier(parameters, data, encoded)
    models['downstream_classifier'] = downstream_classifier
    metrics['classification_metrics'] = classification_metrics
    return encoded, models, metrics


def run_convolutional_autoencoder(parameters, data):
    if parameters['run_hyperparameter_tuning'] is True:
        grid_result = run_hyperparameter_optimization(parameters, data['training_data'])
    convolutional_autoencoder = Convolutional_Autoencoder(parameters)
    convolutional_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate']),
                                      loss=tf.keras.losses.MeanSquaredError())
    callback = [tf.keras.callbacks.ModelCheckpoint(os.path.join(parameters['data_directory'], 'saved_model_weights',
                                                                'epoch_{epoch}.hdf5'),
                                                   save_weights_only=True, save_best_only=True, monitor='val_loss'), ]
    reconstruction_metrics = convolutional_autoencoder.fit(data['training_data'], data['training_data'], epochs=parameters['epochs'],
                                                       batch_size=parameters['batch_size'], verbose=1, shuffle=True, callbacks=callback,
                                                       validation_data=(data['validation_data'], data['validation_data']))
    encoded = dict(
        encoded_training_set=convolutional_autoencoder.encoder.predict(data['training_data']),
        encoded_test_set=convolutional_autoencoder.encoder.predict(data['test_data']),
        encoded_full_set=convolutional_autoencoder.encoder.predict(data['full_data'])
    )
    return encoded, reconstruction_metrics, convolutional_autoencoder


def run_classifier(parameters, data, encoded):
    downstream_classifier = Classifier(parameters)
    downstream_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['classifier_lr']),
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                  metrics=['accuracy'])
    classification_metrics = downstream_classifier.fit(encoded['encoded_training_set'], data['y_train'],
                                                       epochs=parameters['classifier_epochs'], verbose=1, shuffle=True)
    return downstream_classifier, classification_metrics


class Convolutional_Autoencoder(tf.keras.Model):
    def __init__(self, parameters):
        super(Convolutional_Autoencoder, self).__init__()
        self.encoder = Encoder(parameters)
        self.decoder = Decoder(parameters)

    def call(self, inputs):
        out_encoder = self.encoder(inputs)
        out_decoder = self.decoder(out_encoder)
        return out_decoder


class Encoder(tf.keras.Model):
    def __init__(self, parameters):
        super(Encoder, self).__init__()
        # for encoder ---------->
        self.convolutional_1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.convolutional_2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flatten = Flatten()
        self.latent1 = Dense(parameters['latent_dim'])
        self.latent2 = Dense(parameters['latent_dim'])
        # general
        self.flipped = tf.keras.layers.RandomFlip("horizontal")
        self.rotated = tf.keras.layers.RandomRotation(0.1)
        self.dropout = Dropout(parameters['dropout_val'])

    def call(self, inputs, training=False):
        encoder_layers = [self.flipped, self.rotated, self.convolutional_1, self.dropout, self.convolutional_2,
                          self.dropout, self.flatten]
        output = inputs
        for layer in encoder_layers:
            output = layer(output)
        z_mean = self.latent1(output)
        z_log_var = self.latent2(output)
        output = self.sample_from_latent([z_mean, z_log_var])
        return output

    @staticmethod
    def sample_from_latent(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class Decoder(tf.keras.Model):
    def __init__(self, parameters):
        super(Decoder, self).__init__()
        # for decoder ------------------>
        self.convolutional_3 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')
        self.convolutional_4 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.convolutional_5 = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')
        self.decoder_1 = Dense(units=7 * 7 * 32, activation=tf.nn.relu)
        self.decoder_2 = Reshape(target_shape=(7, 7, 32))
        # general
        self.dropout = Dropout(parameters['dropout_val'])

    def call(self, inputs, training=False):
        decoder_layers = [self.decoder_1, self.decoder_2, self.convolutional_3, self.dropout, self.convolutional_4, self.dropout,
                          self.convolutional_5]
        output = inputs
        for layer in decoder_layers:
            output = layer(output)
        return output


class Classifier(tf.keras.Model):
    def __init__(self, parameters):
        super(Classifier, self).__init__()
        self.classifier_1 = Dense(int(np.round(parameters['latent_dim'] / 2)), activation='relu')
        self.classifier_2 = Dense(int(np.round(parameters['latent_dim'] / 4)), activation='relu')
        self.classifier_3 = Dense(7, activation='softmax')

    def call(self, inputs):
        classifier_layers = [self.classifier_1, self.classifier_2, self.classifier_3]
        output = inputs
        for layer in classifier_layers:
            output = layer(output)
        return output


def run_hyperparameter_optimization(parameters, training_data):
    param_grid = dict(batch_size=parameters['tuning_batch_size'], epochs=parameters['tuning_epochs'], learning_rate=parameters['tuning_learning_rate'])
    model = KerasRegressor(build_fn=create_convolutional_autoencoder, verbose=0)
    grid = HalvingGridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(training_data, training_data)
    return grid_result
