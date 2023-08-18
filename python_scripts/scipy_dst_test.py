import numpy as np
from scipy.fft import dst, idst

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

npts = 1024
radius = 10.24

dr = radius / float(npts)
dk = 2 * np.pi / (2 * float(npts) * dr)
T = 1.6
beta = 1 / T
r = np.arange(0.5, npts) * dr
k = np.arange(0.5, npts) * dk
lj = lj(1.0, 1.0, r)
mayer = np.exp(-beta * lj) - 1.0
mayer_r2k2r = idfbt(r, k, dfbt(r, k, mayer, dr), dk)

ones = np.ones(r.shape)

kfromr = dfbt(r, k, ones, dr)

print(mayer)
print(mayer_r2k2r)
