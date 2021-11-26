

__all__ = ["calc_kl", "calc_js","est_pdf", "integrate"]

import numpy as np


def calc_kl(pk, qk):
    '''
    A function to calculate the Kullback-Libler divergence between p and q distribution.
    Arguments:
    pk: pdf values from p distribution;
    qk: pdf values from q distribution.
    '''
    return np.nan_to_num(pk*np.log(pk/qk))


def integrate( bins, dx = 1):
    return np.trapz( bins, dx=dx)


def calc_js(pk, qk):
    '''
    A function to calculate the Jensen-Shanon divergence between p and q distribution.
    Arguments:
    pk: pdf values from p distribution
    qk: pdf values from q distribution
    '''
    mk = 0.5*(pk+qk)
    return 0.5*(calc_kl(pk, mk) + calc_kl(qk, mk))

def est_pdf(hist_counts, beta=1):
    '''
    A function to make pdf estimation using a generalization of Laplace rule. Using that we can avoid bins with zero probability

    This implementation is based on:
    https://papers.nips.cc/paper/2001/file/d46e1fcf4c07ce4a69ee07e4134bcef1-Paper.pdf
    Arguments:
    hist_counts: the histogram counts
    beta: the beta factor for the probability estimation
    '''
    K     = len(hist_counts)
    kappa = K*beta 
    pdf   = (hist_counts + beta)/(hist_counts.sum() + kappa)
    return pdf