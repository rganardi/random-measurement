# vim: foldmethod=marker
#imports    {{{
import numpy as np
from math import sqrt, pi
from cmath import exp
#}}}
def ann(d): #{{{
    #create annihilation operator of dimension d
    a = np.zeros([d, d], dtype=complex)
    for i in range(d-1):
        a[i, i+1] = sqrt(i+1)
    return a
#}}}
def ptrace(rho, dims, mask):    #{{{
    #computes partial trace of an np.array object
    #traces out the subsystems where mask == 1
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
    rho = perm(rho, dims, np.argsort(1 - sp_mask))

    return rho.reshape([dA, dB, dA, dB]).transpose([0,2,1,3]).trace(axis1=0, axis2=1)
#}}}
def perm(rho, dims, order): #{{{
    #permute the subsystems in rho
    dim = 1
    for iter in dims:
        dim = dim*iter
    if len(rho) != dim:
        print("rho is of the wrong shape")
        return
    shape = np.concatenate((dims, dims))
    t_order = np.arange(2*len(dims)).reshape(2,len(dims)).transpose().flatten()
    index = np.arange(2*len(dims)).reshape(len(dims),2)[order].flatten()
    inv_t_order = np.argsort(t_order)
    return rho.reshape(shape).transpose(t_order).transpose(index).transpose(inv_t_order).reshape([dim, dim])
#}}}
def randu(dim): #{{{
    z = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    q, r = np.linalg.qr(z)
    return q@np.diag(r.diagonal().conjugate()/abs(r.diagonal()))
#}}}
def absm(rho): #{{{
    u, s, v = np.linalg.svd(rho)
    return v.conj().transpose() @ np.diag(s) @ v
#}}}
def weyl_x(n):  #{{{
    x = np.zeros([n,n], dtype=complex)
    for i in range(n):
        x[((i+1) % n), i] = 1
    return x
#}}}
def weyl_z(n):  #{{{
    z = np.zeros([n,n], dtype=complex)
    omega = exp(2j*pi/n)
    for i in range(n):
        z[i,i] = omega**i
    return z
#}}}
