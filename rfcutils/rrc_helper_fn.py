#  https://nvlabs.github.io/sionna/examples/Pulse_shaping_basics.html#Pulse-shaping-of-a-sequence-of-QAM-symbols

import sionna as sn


def get_psf(samples_per_symbol, span_in_symbols, beta):
    """
    samples_per_symbol: Number of samples per symbol, i.e., the oversampling factor
    span_in_symbols: Filter span in symbols
    beta: Roll-off factor

    returns pulse-shaped-filter (root-raised-cosine) with the specified parameters
    """
    rrcf = sn.signal.RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
    return rrcf


def matched_filter(sig, samples_per_symbol, span_in_symbols, beta):
    """
    sig: input signal
    samples_per_symbol: Number of samples per symbol, i.e., the oversampling factor
    span_in_symbols: Filter span in symbols
    beta: Roll-off factor

    Applies RRC matched-filter on input signal
    """
    rrcf = get_psf(samples_per_symbol, span_in_symbols, beta)
    # apply the matched filter
    x_mf = rrcf(sig, padding="same")
    return x_mf
