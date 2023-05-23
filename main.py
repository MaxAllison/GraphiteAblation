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
    for z in Zx:
        y = integrate.quad(slice, 0, z)[0] * 10 ** 8  # Scipy numerical integration
        res[i] = y # Append to the list of volumes
        i += 1
    return res

def PeakFluence(E,w0):
    """
    Calculates the peak laser fluence as a function of energy (E) and beam focus
    radius (w0) ie the radius at which intensity falls to 1/exp(2)
    Args:
        E: Pulse energy (J)
        w0: Beam focus radius (cm)
    Returns:
        Peak laser fluence (J/cm^2)
    """
    return 2*E / (m.pi * (w0**2))

def FluenceGaussian(E,w0,r):
    """
    Calculates fluence distribution as a function of energy (E), beam focus
    radius (w0), and radius to the beam centre (r)
    Args:
        E: Pulse energy (J)
        w0: Beam focus radius (cm)
        r: Radius to beam centre (cm)
    Returns:
        Peak laser fluence (J/cm^2)
    """
    F_c = PeakFluence(E,w0)
    return F_c * m.exp(-(2*r**2)/w0**2)

def Depth2(E, w0, F_tr, r):
    """
    Single pulse depth
    Args:
        E: Pulse energy (J)
        w0: Beam focus radius (cm)
        F_tr: Threshold fluence (J/cm^2)
        r: Radius to beam centre (cm)
    Returns:
        Ablated depth (cm)
    """
    Fc = PeakFluence(E,w0)
    return zmax(Fc, F_tr) - 2 * (r**2) / (w0**2)

def Volume2(E, w0, F_tr):
    Fc = PeakFluence(E,w0)
    return ((m.pi * w0**2) / (4 * a)) * zmax(Fc, F_tr)**2

OptimalFluence = F_tr*m.exp(1)**2 /2
OptimalVolume = m.pi*w0**2 /a
def OptimalBeamWaist(E):
    return 2*E/(F_tr*m.pi*m.exp(1)**2)

# Arrays for plots
Fx = np.arange(F_tr-0.01, F_0+0.13, 0.01) # Array of fluences from 0 to 0.43 (J/cm^2)
Zx = np.arange(0, int(zmax(F_0, F_tr) * 10 ** 7), 5) # List of depths from 0 to zmax (nm)
Rx = (10**-9) * np.arange(-1,1, 0.01) # Array of radii, distance to centre. (cm)

def Energy(radius, Fluence):
    """
    Energy as area*fluence
    Args:
        Fluence: Fluence (J/cm^2)
        radius: Radius to beam centre (cm)
    Returns:
        Energy (J)
    """
    return m.pi*(radius**2) * Fluence

def Number(F):
    E = 64* 10**-6 # 64 microJoules
    rho = cdensity(F)
    vol = Volume2(E, w0, F_tr)
    return rho*vol

# Plot Diameter vs Depth
plt.figure(figsize=(8, 6))
plt.scatter(Zx, np.vectorize(diameter)(Zx, F_0, F_tr) * 10 ** 7, s=10, label="Diameter")
plt.axvline(zmax(F_0, F_tr) * 10 ** 7, color="red", linestyle="--", label=f"zmax = {round(zmax(F_0, F_tr) * 10 ** 7, 2)} nm")
plt.title("Diameter vs Depth")
plt.xlabel("Depth (nm)")
plt.ylabel("Diameter (nm)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Volume vs Depth
plt.figure(figsize=(8, 6))
plt.scatter(Zx, volume(Zx)/100, s=10, label="Volume")
plt.axvline(zmax(F_0, F_tr) * 10 ** 7, color="red", linestyle="--", label=f"zmax = {round(zmax(F_0, F_tr) * 10 ** 7, 2)} nm")
plt.title("Volume vs Depth")
plt.xlabel("Depth (nm)")
plt.ylabel("Volume (cm^3)")
plt.legend()
plt.grid(True)
plt.show()

""" Plot Slice vs Depth
# plt.figure(figsize=(8, 6))
# plt.scatter(Zx, np.vectorize(slice)(Zx), s=10, label="Slice")
# plt.axvline(zmax(F_0, F_tr) * 10 ** 7, color="red", linestyle="--", label=f"zmax = {round(zmax(F_0, F_tr) * 10 ** 7, 2)} nm")
# plt.title("(Optional) Slice vs Depth")
# plt.xlabel("Depth (nm)")
# plt.ylabel("Slice (100 cm^3)")
# plt.legend()
# plt.grid(True)
# plt.show()
"""
# Plot Density vs Fluence
plt.figure(figsize=(8, 6))
plt.scatter(Fx, np.vectorize(cdensity)(Fx), s=10, label="Density")
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

# Plot Depth1 (zmax) vs Fluence
plt.figure(figsize=(8, 6))
plt.scatter(Fx, np.vectorize(zmax)(Fx, F_tr), s=10, label="Density")
plt.axvline(F_0, linestyle = "-.")
plt.text(0.285,-0.000075,'Peak laser Fluence',rotation=90)
plt.axvline(F_tr, linestyle = "-")
plt.text(0.115,-0.000100,'Treshold Fluence',rotation=90)
plt.title("Depth1 (zmax) vs Fluence")
plt.xlabel("Fluence (J/m^2)")
plt.ylabel("Depth/zmax (cm^2)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Depth2 vs Distance to centre for different Fluences
StepFluence = np.sort(np.concatenate((np.arange(0.1, 1, 0.1), np.array([0.13,0.3])), axis=0))
plt.figure(figsize=(8, 6))
for F in StepFluence:
    plt.scatter(Rx, np.vectorize(Depth2)(Energy(Rx,F), w0, F_tr, Rx), s=10, label=f'{F}')
plt.title("Depth2 vs Distance to centre for different Fluences")
plt.xlabel("Distance to centre (nm)")
plt.ylabel("Depth (cm^2)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Volume2 vs Fluence
plt.figure(figsize=(8, 6))
plt.scatter(Fx, np.vectorize(Volume2)(np.vectorize(Energy)(2*w0, Fx), w0, F_tr))
plt.title("Volume2 vs Fluence")
plt.xlabel("Fluence (J/cm^2)")
plt.ylabel("Volume (cm^3)")
plt.grid(True)
plt.show()

# Plot Number of Carriers vs Fluence
plt.figure(figsize=(8, 6))
plt.scatter(Fx, Number(Fx))
plt.title("Number vs Fluence")
plt.xlabel("Fluence (J/cm^2)")
plt.ylabel("Number of carriers")
plt.axvline(F_0, linestyle = "-.")
plt.text(0.285, Number(np.average(Fx)),'Peak laser Fluence',rotation=90)
plt.axvline(F_tr, linestyle = "-")
plt.text(0.115, Number(np.average(Fx)),'Treshold Fluence',rotation=90)
plt.grid(True)
plt.show()


