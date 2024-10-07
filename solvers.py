# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:46:59 2023

@author: uqalim8
"""

import torch
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# VERBOSE = True
STAT = ("ite", "|rk|/|b|", "|Ark|/|Ab|")

def print_stats(stats, *args):
    items = "{:^5}" + (len(stats) - 1) * " | {:^10}"
    statistics = "{:^5}" + (len(stats) - 1) * " | {:^10.2e}"
    if not (args[0] - 1) % 10:
        print(len(stats) * 12 * "-" + "-")
        print(items.format(*stats))
        print(len(stats) * 12 * "-" + "-")
    print(statistics.format(*args))
    
def minres(A, b, rtol = 1e-7, maxit = None, reO = False, VERBOSE=True):
    
    if maxit is None:
        maxit = A.shape[0]
    
    betak = np.linalg.norm(b)
    norm_b = betak
    rkm1 = b
    vk = b / betak
    v1 = vk
    vkm1, xkm1, dkm1, dkm2 = 4 * [np.zeros_like(b)]
    ckm1, skm1, phikm1 = -1, 0, betak
    delta1k, epsk, k = 0, 0, 0
    
    if reO:
        V = vk.reshape(-1, 1)
        
    abnorm = np.linalg.norm(Avec(A, b))
    arnorm = np.linalg.norm(Avec(A, rkm1))
    
    while k < maxit and (arnorm / abnorm) > rtol:
        k += 1
        pk = Avec(A, vk)
        # alphak = np.dot(vk, pk)
        alphak = vk.transpose() @ pk
        pk = pk - betak * vkm1
        pk = pk - alphak * vk
        betakp1 = np.linalg.norm(pk)
        vkp1 = pk / betakp1
        
        if reO:
            #vkp1 = vkp1 - V @ (Avec(V.T, vkp1))
            for i in range(V.shape[-1]):
                # vkp1 = vkp1 - V[:, i] * np.dot(V[:, i], vkp1) 
                vkp1 = vkp1 - V[:, i] * (V[:, i].transpose() @ vkp1)
            V = np.concatenate([V, vkp1.reshape(-1, 1)], dim = 1)
            
        delta2k = ckm1 * delta1k + skm1 * alphak
        gamma1k = skm1 * delta1k - ckm1 * alphak
        epskp1 = skm1 * betakp1
        delta1kp1 = -ckm1 * betakp1
        
        gamma2k = np.sqrt(gamma1k ** 2 + betakp1 ** 2)
        
        if VERBOSE:
            print_stats(STAT, k, np.linalg.norm(rkm1) / norm_b, arnorm / abnorm)
            
        if ckm1 * gamma1k >= 0:
            print("NPC detected")
            return xk
            
        if gamma2k > 0:
            ck = gamma1k / gamma2k
            sk = betakp1 / gamma2k
            tauk = ck * phikm1
            phik = sk * phikm1
            dk = (vk - delta2k * dkm1 - epsk * dkm2) / gamma2k
            xk = xkm1 + tauk * dk
            if betakp1 > 0:
                rk = sk ** 2 * rkm1 - phik * ck * vkp1
            else:
                rk = np.zeros_like(b)
                print("the exact solution has been obtained")
                return xk
            
        else:
            ck, sk, tauk, phik = 0, 1, 0, phikm1
            rk = rkm1
            xk = xkm1
            print("a solution to the normal equation has been obtained")
            return xk
        
        # update
        epsk = epskp1
        xkm1 = xk
        vkm1 = vk
        vk = vkp1
        betak = betakp1
        delta1k = delta1kp1
        ckm1 = ck
        skm1 = sk
        phikm1 = phik
        dkm2 = dkm1
        dkm1 = dk
        rkm1 = rk
        arnorm = np.linalg.norm(Avec(A, rkm1))
        
    if VERBOSE:
        print_stats(STAT, k + 1, np.linalg.norm(rkm1) / norm_b, arnorm / abnorm)
        
    return xk
        
def CG(A, b, rtol, maxit = None, reO = False, VERBOSE=True):

    if maxit is None:
        maxit, _ = A.shape

    xk = torch.zeros_like(b, dtype = torch.float64)
    r = b - Avec(A, xk)
    pk = r.clone()
    norm_r = torch.norm(r)
    norm_r0 = norm_r
    
    i = 0
    Ap = Avec(A, pk)
    pAp = torch.dot(pk, Ap)
        
    alpha, beta = 0, 0
    while i < maxit and norm_r / norm_r0 > rtol:
        i += 1
        alpha = (norm_r ** 2) / pAp
        
        if VERBOSE:
            print_stats(STAT, i, norm_r / norm_r0, 0)
        
        # standard CG
        xk = xk + alpha * pk
        r = r - alpha * Ap               #r = b - Avec(A, xk) # try another way to evaluate residual
        norm_rk = torch.norm(r)
        beta = (norm_rk / norm_r) ** 2   #beta = -torch.dot(r, Ap) / pAp # original beta
        pk = r + beta * pk
        
        # update
        Ap = Avec(A, pk)
        pAp = torch.dot(pk, Ap)
        norm_r = norm_rk
        
    if VERBOSE:
        print_stats(STAT, i + 1, norm_r / norm_r0, 0)
        
    return xk

def CR(A, b, rtol, init = False, x0 = 0, maxit = None, reO = True, VERBOSE=True):
    
    if maxit is None:
        maxit, _ = A.shape

    if init:
        xk = x0
    else:
        xk = np.zeros_like(b, dtype = np.float64)
    
    r = b - Avec(A, xk)
    norm_r0 = np.linalg.norm(r)
    Ar = Avec(A, r)
    # rAr = torch.dot(r, Ar)
    rAr = r.transpose() @ Ar
    # p = r.clone()
    p = np.copy(r)
    # Ap = Ar.clone()
    Ap = np.copy(Ar)
    i, beta = 0, 0
    
    norm_Ar0 = np.linalg.norm(Ap)
    while i < maxit and np.linalg.norm(Ar) / norm_Ar0 > rtol:
        i += 1
        # pAAp = torch.dot(Ap, Ap)
        pAAp = Ap.transpose() @ Ap
        
        if VERBOSE:
            print_stats(STAT, i, np.linalg.norm(r) / norm_r0, np.linalg.norm(Ar) / norm_Ar0)
        
        alpha = rAr / pAAp
        
        if rAr < 0:
            print("NPC detected, it {}".format(i))
            return xk, i, r, np.linalg.norm(r) / norm_r0, np.linalg.norm(Ar) / norm_Ar0
        
        xk = xk + alpha * p
        r = r - alpha * Ap
        Ar = Avec(A, r)
        # rArp = torch.dot(r, Ar)
        rArp = r.transpose() @ Ar
        beta = rArp / rAr
        p = r + beta * p
        Ap = Ar + beta * Ap
        
        # update
        rAr = rArp
        
    if VERBOSE:
        print_stats(STAT, i + 1, np.linalg.norm(r) / norm_r0, np.linalg.norm(Ar) / norm_Ar0)
        
    return xk, i, r, np.linalg.norm(r) / norm_r0, np.linalg.norm(Ar) / norm_Ar0

def CR_torch(A, b, rtol, init = False, x0 = 0, maxit = None, VERBOSE=True):
    '''
    INPUT:
        A: square nxn matrix
        b: coloumn vector size (n)
    '''


    if maxit is None:
        maxit, _ = A.shape

    if init:
        xk = x0
    else:
        xk = torch.zeros_like(b, dtype = torch.float64, device=device)


    def Avec(A, x):
        if callable(A):
            return A(x)
        return torch.matmul(A, x)
    
    r = b - Avec(A, xk)
    norm_r0 = torch.norm(r)
    Ar = Avec(A, r)
    rAr = torch.dot(r, Ar)
    p = r.clone()
    Ap = Ar.clone()
    i, beta = 0, 0
    
    norm_Ar0 = torch.norm(Ap)
    while i < maxit and torch.norm(Ar) / norm_Ar0 > rtol:
        i += 1
        pAAp = torch.dot(Ap, Ap)
        # pAAp = Ap.transpose() @ Ap
        
        if VERBOSE:
            print_stats(STAT, i, torch.norm(r) / norm_r0, torch.norm(Ar) / norm_Ar0)
        
        alpha = rAr / pAAp
        
        if rAr < 0:
            print("NPC detected, it {}".format(i))
            return xk, i, r, torch.norm(r) / norm_r0, torch.norm(Ar) / norm_Ar0
        
        xk = xk + alpha * p
        r = r - alpha * Ap
        Ar = Avec(A, r)
        rArp = torch.dot(r, Ar)
        # rArp = torch.matmul(r.transpose(0,1),Ar)
        # rArp = r.transpose() @ Ar
        beta = rArp / rAr
        p = r + beta * p
        Ap = Ar + beta * Ap
        
        # update
        rAr = rArp
        
    if VERBOSE:
        print_stats(STAT, i + 1, torch.norm(r) / norm_r0, torch.norm(Ar) / norm_Ar0)
        
    return xk, i, r, torch.norm(r) / norm_r0, torch.norm(Ar) / norm_Ar0


def Avec(A, x):
    if callable(A):
        return A(x)
    return np.dot(A, x)


# if "__main__" == __name__:
#     #torch.manual_seed(1234)
    
#     N = 1000
#     MAXIT = 1000

#     D = torch.diag(torch.randn(N, dtype = torch.float64) + 0.1)
#     b = torch.randn(N, dtype = torch.float64)
    
#     print("\nMINRES\n")
#     xMR = minres(D, b, 1e-6, MAXIT, reO = True)
    
#     print("\nCR\n")
#     xCR = CR(D, b, 1e-6, MAXIT)
    
#     print("\nCG\n")
#     xCG = CG(D, b, 1e-6, MAXIT)



