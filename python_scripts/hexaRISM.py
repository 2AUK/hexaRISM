import numpy as np
from scipy.fft import dst, idst
import matplotlib.pyplot as plt

def lj(eps, sig, r):
    return 4.0 * eps * (np.power(sig / r, 12.0) - np.power(sig / r, 6.0))

def dfbt(
    r: np.ndarray, k: np.ndarray, fr: np.ndarray, dr: float
) -> np.ndarray:
    constant = 2 * np.pi * dr / k
    return constant * dst(fr * r, type=4)


def idfbt(
    r: np.ndarray, k: np.ndarray, fk: np.ndarray, dk: float
) -> np.ndarray:
    constant = dk / (4.0 * np.pi * np.pi) / r
    return constant * dst(fk * k, type=4)

def j0(a):
    return np.sin(a) / a

def RISM(ck, wk, p):
    return ((ck * wk * wk) / (1.0 - 6.0 * p * ck * wk)) - ck

def hnc(tr, ur, beta):
    return np.exp(-beta * ur + tr) - 1.0 - tr

npts = 1024
radius = 10.24

dr = radius / float(npts)
dk = 2 * np.pi / (2 * float(npts) * dr)
T = 1.6
p = 0.2
kb = 1.0
beta = 1 / T / kb
l = 1.0

r = np.arange(0.5, npts) * dr
k = np.arange(0.5, npts) * dk

lj_potential = lj(1.0, 1.0, r)

j0_adj = 2.0 * j0(k * l)
j0_bet = 2.0 * j0(k * l * np.sqrt(3.0))
j0_opp = j0(2.0 * k * l)
wk = 1.0 + j0_adj + j0_bet + j0_opp 

cr = np.zeros_like(r)
tr = np.zeros_like(r)

max_iter, tol = 10, 1e-7
damp = 1.0

for i in range(max_iter):
    print("Iteration {i}".format(i=i))
    c_prev = cr.copy()
    ck = dfbt(r, k, cr, dr)
    tk = RISM(ck, wk, p)
    tr = idfbt(r, k, tk, dk)
    c_a = hnc(tr, lj_potential, beta)
    cr = c_prev + (damp * (c_a - c_prev))
    print(cr)
    
