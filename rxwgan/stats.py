

__all__ = ["calc_kl", "calc_js","est_pdf", "integrate","calculate_divergences", "calculate_l1_and_l2_norm_errors", "eps"]

import numpy as np
import itertools

eps = 1e-3

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


#
# Calculate kl pair to pair using permutation between real and fake samples
#
def calculate_divergences( real_samples, fake_samples ):
    kl = []; js = []; l1 = []
    for r_idx, f_idx in itertools.permutations(list(range(real_samples.shape[0])), 2):
        # Int_inf_to_plus_int p(x)/(dx*total) * dx = 1
        real_pdf, bins = np.histogram( real_samples[r_idx].flatten(), bins=100, range=(0,1), density=True )
        fake_pdf, bins = np.histogram( fake_samples[f_idx].flatten(), bins=100, range=(0,1), density=True )
        kl.append( integrate( calc_kl(pk = real_pdf + eps, qk = fake_pdf + eps), dx=1/100 ) )
        js.append( integrate( calc_js(pk = real_pdf + eps, qk = fake_pdf + eps), dx=1/100 ) )
    return kl, js

#
# Calculate l1/l2 pair to pair using permutation between real and fake samples
#
def calculate_l1_and_l2_norm_errors(real_samples, fake_samples):
    l1 = []; l2 = []
    for r_idx, f_idx in itertools.permutations(list(range(real_samples.shape[0])), 2):
        # height x width
        hw = len(real_samples[r_idx]) * len(real_samples[r_idx][0])
        # calculate l1
        l1.append( sum(sum( abs( real_samples[r_idx] - fake_samples[f_idx] ) ) )[0] / hw )
        # calculate l2
        # l2_norm_error = 1/HW * Sum ( (y-yhat)**2 ) 
        l2.append( sum(sum( np.power(real_samples[r_idx] - fake_samples[f_idx],2) ) )[0] / hw )

    return l1,l2