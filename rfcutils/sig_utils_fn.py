import numpy as np

# Pow = (1/len(s))*|s|^2
get_pow = lambda s: np.mean(np.abs(s) ** 2)


def get_sinr(s, i, units='dB'):
    """
    s: signal
    i: interference

    returns the SNIR in dB unless specified otherwise
    """
    sinr = get_pow(s) / get_pow(i)
    if units == 'dB':
        return 10 * np.log10(sinr)
    return sinr
