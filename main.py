import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
import pylab as pl
import math as m

# Constants
R = 0.3  # Reflectivity
a = 4.56 * (10 ** 4)  # Absorption coefficient (cm^-1)
h = 6.62 * (10 ** (-34))  # Planck's constant (J/Hz or J.s)
c = 2.998 * (10 ** 10)  # Speed of light (cm/s)

# Parameters
F_0 = 0.30  # Peak laser fluence (J/cm^2)
F_tr = 0.13  # Threshold fluence for critical damage (J/cm^2)
w = 1030 * (10 ** -7)  # Wavelength (cm)
mu = c / w
sigma = 2 * 1030 * (10 ** -7)  # Beam waist (cm)
w_0 = 8.2 * 10**-4 # beam focus radius (cm)

def cdensity(F: float):
    """
    Calculates charge carrier density as a function of fluence.
    Args:
        F: Fluence in J/cm^2
    Returns:
        Charge carrier density in cm^-3
    """
    return ((1 - R) * F * a) / (h * mu)

F = np.arange(0, F_0+0.13, 0.02)

def diameter(z, F_0, F_tr):
    """
    Calculates the diameter as a function of depth (z).
    Args:
        z: Depth in nm (converted to cm)
        F_0: Peak laser fluence (J/cm^2)
        F_tr: Threshold fluence of the ion emission (J/cm^2)
    Returns:
        Diameter in cm
    """
    z = z * 10 ** -7  # Convert depth from nm to cm
    return sigma * m.sqrt(m.log(F_0 * m.exp(-a * z) / F_tr))

def zmax(F_0, F_tr):
    """
    Calculates the maximum depth (zmax) based on the given parameters.
    Args:
        F_0: Peak laser fluence (J/cm^2)
        F_tr: Threshold fluence for critical damage (J/cm^2)
    Returns:
        Maximum depth (zmax) in cm
    """
    return (m.log(F_0 / F_tr)) / a

def slice(x):
    """
    Named slice because a volume integral is an sum of 3d slices.
    Calculates the slice (integrand) value at a given depth (x).
    Args:
        x: Depth in nm
    Returns:
        Slice value
    """
    return ((diameter(x, F_0, F_tr) / 2) ** 2) * m.pi

Z = np.arange(0, int(zmax(F_0, F_tr) * 10 ** 7), 5)  # List of depths in nm

def volume(x):
    """
    Calculates the volume as a function of depth (x) using numerical integration.
    Args:
        x: Depth in nm
    Returns:
        Volume in 100 cm^3
    """
    res = np.zeros_like(x)
    i = 0
    for z in Z:
        y = integrate.quad(slice, 0, z)[0] * 10 ** 8  # Scipy numerical integration
        res[i] = y # Append to the list of volumes
        i += 1
    return res


def PeakFluence(E,w_0):
    """
    Calculates the peak laser fluence as a function of energy (E) and beam focus
    radius (w_0) ie the radius at which intensity falls to 1/exp(2)
    Args:
        E: Pulse energy (J)
        w_0: Beam focus radius (cm)
    Returns:
        Peak laser fluence (J/cm^2)
    """
    return 2*E / (m.pi * (w_0**2))

def FluenceGaussian(E,w_0,r):
    """
    Calculates fluence distribution as a function of energy (E), beam focus
    radius (w_0), and radius to the beam centre (r)
    Args:
        E: Pulse energy (J)
        w_0: Beam focus radius (cm)
        r: Radius to beam centre (cm)
    Returns:
        Peak laser fluence (J/cm^2)
    """
    F_c = PeakFluence(E,w_0)
    return F_c * m.exp(-(2*r**2)/w_0**2)

# Plot Diameter vs Depth
plt.figure(figsize=(8, 6))
plt.scatter(Z, np.vectorize(diameter)(Z, F_0, F_tr) * 10 ** 7, s=10, label="Diameter")
plt.axvline(zmax(F_0, F_tr) * 10 ** 7, color="red", linestyle="--", label=f"zmax = {round(zmax(F_0, F_tr) * 10 ** 7, 2)} nm")
plt.title("Diameter vs Depth")
plt.xlabel("Depth (nm)")
plt.ylabel("Diameter (nm)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Volume vs Depth
plt.figure(figsize=(8, 6))
plt.scatter(Z, volume(Z)/100, s=10, label="Volume")
plt.axvline(zmax(F_0, F_tr) * 10 ** 7, color="red", linestyle="--", label=f"zmax = {round(zmax(F_0, F_tr) * 10 ** 7, 2)} nm")
plt.title("Volume vs Depth")
plt.xlabel("Depth (nm)")
plt.ylabel("Volume (cm^3)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Slice vs Depth
plt.figure(figsize=(8, 6))
plt.scatter(Z, np.vectorize(slice)(Z), s=10, label="Slice")
plt.axvline(zmax(F_0, F_tr) * 10 ** 7, color="red", linestyle="--", label=f"zmax = {round(zmax(F_0, F_tr) * 10 ** 7, 2)} nm")
plt.title("(Optional) Slice vs Depth")
plt.xlabel("Depth (nm)")
plt.ylabel("Slice (100 cm^3)")
plt.legend()
plt.grid(True)
plt.show()

# Plot density vs Fluence
plt.figure(figsize=(8, 6))
plt.scatter(F, np.vectorize(cdensity)(F), s=10, label="Density")
plt.axvline(F_0, linestyle = "-.")
plt.text(0.285,0,'Peak laser Fluence',rotation=90)
plt.axvline(F_tr, linestyle = "-")
plt.text(0.115,0,'Treshold Fluence',rotation=90)
plt.title("Carrier density vs Fluence")
plt.xlabel("Fluence (J/m^2)")
plt.ylabel("Density (cm^3) or Number per cm^3")
plt.legend()
plt.grid(True)
plt.show()


