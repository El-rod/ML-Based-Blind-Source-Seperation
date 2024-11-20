import sionna as sn
import numpy as np
import tensorflow as tf

from .rrc_helper_fn import get_psf, matched_filter


# Binary source to generate uniform i.i.d. bits (random binary tensors)
binary_source = sn.utils.BinarySource()

samples_per_symbol = 16
span_in_symbols = 8
beta = 0.5

# 4-QAM constellation
NUM_BITS_PER_SYMBOL = 2
constellation = sn.mapping.Constellation("qam",
                                         NUM_BITS_PER_SYMBOL,
                                         trainable=False)
                                # trainable is false by default so...

# Mapper: maps binary tensors to points of a constellation.
mapper = sn.mapping.Mapper(constellation=constellation)
# Demapper: computes log-likelihood ratios (LLRs) or hard-decisions on bits for a tensor of received symbols.
demapper = sn.mapping.Demapper("app",
                               constellation=constellation)

# AWGN channel:
# complex AWGN noise with variance N0 to the input.
# The noise has variance N0/2 per real dimension
awgn_channel = sn.channel.AWGN()


#
def generate_qpsk_signal(batch_size, num_symbols, ebno_db=None):
    """
    batch_size: how many
    num_symbols: number of symbols
    ebno_db: energy per bit to noise power spectral density ratio
    """
    bits = binary_source([batch_size, num_symbols * NUM_BITS_PER_SYMBOL])  # Blocklength
    return modulate_qpsk_signal(bits, ebno_db)


def qpsk_matched_filter_demod(sig, no=1e-4, soft_demod=False):
    """
    sig: signal (the received symbols)
    no: N0 – noise variance estimate
    soft_demod: ???
    """
    # x after matched filter
    x_mf = matched_filter(sig, samples_per_symbol, span_in_symbols, beta)
    # get number of symbol from symbol length to signal length
    num_symbols = sig.shape[-1] // samples_per_symbol
    # downsamples a tensor along a specified axis by retaining one out of samples_per_symbol elements.
    ds = sn.signal.Downsampling(samples_per_symbol, samples_per_symbol // 2, num_symbols)
    x_hat = ds(x_mf)
    # x / (samples_per_symbol)^0.5 – complex type
    x_hat = x_hat / tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))
    # log likelihood ratio
    llr = demapper([x_hat, no])
    # demodulation
    # TO UNDERSTAND
    if soft_demod:
        return llr, x_hat
    return tf.cast(llr > 0, tf.float32), x_hat


def modulate_qpsk_signal(info_bits, ebno_db=None):
    """
    info_bits:
    ebno_db: energy per bit to noise power spectral density ratio
    """
    # map info_bits to points of a constellation.
    x = mapper(info_bits)
    # upsamples x and pads with zeros
    us = sn.signal.Upsampling(samples_per_symbol)
    x_us = us(x)
    x_us = tf.pad(x_us, tf.constant([[0, 0, ], [samples_per_symbol // 2, 0]]), "CONSTANT")
    x_us = x_us[:, :-samples_per_symbol // 2]
    # apply RRC matched-filter on x
    x_rrcf = matched_filter(x_us, samples_per_symbol, span_in_symbols, beta)
    if ebno_db is None:
        y = x_rrcf
    # if there is noise add it to x
    else:
        # compute the noise variance No for a given Eb/No in dB.
        no = sn.utils.ebnodb2no(ebno_db=ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)  # Coderate set to 1 as we do uncoded transmission here
        y = awgn_channel([x_rrcf, no])
    y = y * tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))
    return y, x, info_bits, constellation

