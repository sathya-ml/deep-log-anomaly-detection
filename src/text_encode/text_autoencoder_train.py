import logging

import tensorflow as tf

from src import config_loader
from text_autoencoder import utils, autoencoder
from text_autoencoder.train_autoencoder import load_or_create_embeddings, show_parameter_count

_LOGGER = logging.getLogger(__name__)


class TextEmbeddingAutoencoderParameters(object):
    def __init__(
            self, save_dir, embedding_size, lstm_units, learning_rate, batch_size,
            num_epochs, dropout_keep, interval, bidirectional, train_embeddings, embeddings,
            vocab, train, valid
    ):
        self._save_dir = save_dir
        self._embedding_size = embedding_size
        self._lstm_units = lstm_units
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._dropout_keep = dropout_keep
        self._interval = interval
        self._bidirectional = bidirectional
        self._train_embeddings = train_embeddings
        self._embeddings = embeddings
        self._vocab = vocab
        self._train = train
        self._valid = valid

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def lstm_units(self):
        return self._lstm_units

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def dropout_keep(self):
        return self._dropout_keep

    @property
    def interval(self):
        return self._interval

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def train_embeddings(self):
        return self._train_embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def vocab(self):
        return self._vocab

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid


def train_log_message_encoder_main(embedding_autoencoder_args: TextEmbeddingAutoencoderParameters):
    sess = tf.Session()
    wd = utils.WordDictionary(embedding_autoencoder_args.vocab)
    embeddings = load_or_create_embeddings(embedding_autoencoder_args.embeddings, wd.vocabulary_size,
                                           embedding_autoencoder_args.embedding_size)

    _LOGGER.info('Reading training data')
    train_data = utils.load_binary_data(embedding_autoencoder_args.train)
    _LOGGER.info('Reading validation data')
    valid_data = utils.load_binary_data(embedding_autoencoder_args.valid)
    _LOGGER.info('Creating model')

    train_embeddings = embedding_autoencoder_args.train_embeddings if embedding_autoencoder_args.embeddings else True
    model = autoencoder.TextAutoencoder(embedding_autoencoder_args.lstm_units,
                                        embeddings, wd.eos_index,
                                        train_embeddings=train_embeddings,
                                        bidirectional=embedding_autoencoder_args.bidirectional)

    sess.run(tf.global_variables_initializer())
    show_parameter_count(model.get_trainable_variables())
    _LOGGER.info('Initialized the model and all variables. Starting training.')
    model.train(sess, embedding_autoencoder_args.save_dir, train_data, valid_data,
                embedding_autoencoder_args.batch_size,
                embedding_autoencoder_args.num_epochs, embedding_autoencoder_args.learning_rate,
                embedding_autoencoder_args.dropout_keep, 5.0, report_interval=embedding_autoencoder_args.interval)


if __name__ == '__main__':
    configs = config_loader.initialize_program()
    train_log_message_encoder_main(configs.text_autoencoder_train)
