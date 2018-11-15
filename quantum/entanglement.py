# vim: foldmethod=marker
#imports    {{{
import numpy as np
from . import utilities
#}}}
def negativity(rho, dims, mask):    #{{{
    rho_pt = ptranspose(rho, dims, mask)
    rho_pt_eigvals = np.linalg.eigvalsh(rho_pt)
    return (sum(abs(rho_pt_eigvals)) - rho.trace()).real/2
#}}}
def ptranspose(rho, dims, mask):    #{{{
    #computes the partial transpose of an np.array object
    #transposes the subsystems where mask == 1
    dA = 1
    sp_dims = np.array(dims)
    sp_mask = np.array(mask)
    for iter in sp_dims*sp_mask:
        if iter != 0:
            dA = dA*iter
        else:
            continue

    dB = int((len(rho))/dA)
    if dA*dB != len(rho):
        print("dA doesn't divide the dimension of rho")
        return
    if not ((sp_mask == 0) + (sp_mask == 1)).all():
        print("mask should be either 0 or 1")

    order = np.argsort(1 - sp_mask)
    rho = utilities.perm(rho, dims, order)

    rho_pt = rho.reshape([dA, dB, dA, dB]).transpose([2,1,0,3]).reshape([len(rho), len(rho)])
    return utilities.perm(rho_pt, dims, np.argsort(order))
#}}}
