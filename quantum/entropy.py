# vim: foldmethod=marker
#imports    {{{
import numpy as np
from . import utilities
#}}}
def relative_entropy(rho, sigma, eps=1e-10):    #{{{
    from math import log2
    if not ((np.allclose(rho.conj().transpose(), rho, atol=eps))
            or (np.allclose(sigma.conj().transpose(), sigma, atol=eps))):
        print("rho or sigma is not hermitian")
        return

    [rvals, rvecs] = np.linalg.eigh(rho)
    [svals, svecs] = np.linalg.eigh(sigma)
    rvecs = rvecs.transpose()
    svecs = svecs.transpose()

    if (rvals < -eps).any() or (svals < -eps).any():
        print("rho or sigma is not positive")
        return

    slogvals = []
    for i in svals:
        if abs(i) > eps:
            slogvals.append(log2(i))
        else:
            slogvals.append(0)

    rlogvals = []
    rel_trace = 0
    for i in rvals:
        if abs(i) > eps:
            rel_trace += i*log2(i)

    for i in range(len(rvals)):
        for j in range(len(slogvals)):
            if abs(svals[j]) < eps and norm(rvals[i] * vdot(rvecs[i],svecs[j])) > eps:
                return float('inf')
            else:
                rel_trace -= np.real(rvals[i] * slogvals[j] * np.linalg.norm(np.vdot(rvecs[i],svecs[j])**2 ))

    return rel_trace
#}}}
def mutual_information(rho, dims, mask):    #{{{
    #computes mutual entropy of subsystems with mask 0 and mask 1
    rhoA = utilities.ptrace(rho, dims, mask)
    rhoB = utilities.ptrace(rho, dims, [1-i for i in mask])
    return entropy(rhoA) + entropy(rhoB) - entropy(rho)
#}}}
def entropy(rho):   #{{{
    #computes the entropy of rho
    return -1 * relative_entropy(rho, np.eye(len(rho)))
#}}}
