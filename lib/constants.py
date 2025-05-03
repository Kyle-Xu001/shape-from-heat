"""
Constants for shape from heat reconstruction.

This module provides constants and configuration parameters for:
- Camera settings
- Optimization parameters
- Physical constants
"""

from typing import Dict, Any
import numpy as np

CAMERA = 'Boson'
# CAMERA = 'Boson-Colocated'
FOCUS_DISTANCE: float = 368.3
HFOV: float = 24.3  # degrees
VFOV: float = 19.5  # degrees
IMAGE_WIDTH_METRIC = 2*FOCUS_DISTANCE * \
    np.tan(np.deg2rad(HFOV/2))  # 158.58582759
IMAGE_HEIGHT_METRIC = 2*FOCUS_DISTANCE * \
    np.tan(np.deg2rad(VFOV/2))  # 127.2686625

SIGMA = 5.6704e-8/(1000**2) # W/(mm^2*K^4)
# SIGMA = 5.6704e-8 # W/(m^2*K^4)
EPSILON = 0.90
AMBIENT_TEMP: float = 295.372  # Kelvin
OBJECT_TEMP = AMBIENT_TEMP

# Just took this value from google for a room at 25c. The typical value ranges from 10 to 100
CONVECTION_COEFF = 10/(1000**2) # W/(mm^2*K)

# Camera Radiometric Calibration as a Linear Function
# y = mx + c
# y = temperature in Kelvin
# x = raw value from the camera
# P1 = slope, P2 = intercept
# P1 = 8.71124796e-03
# P2 = 1.02891960e+02

## Fitting after vignetting correction of blackbody videos
P1 = 8.69660008e-03
P2 = 1.03033911e+02


# Density of the material in kg/mm^3 at 98 celcius
Density = {
    "aluminium": 2700/(1000**3),
    "pvc": 1330/(1000**3),
    "glass": 2500/(1000**3),
    "copper": 8960/(1000**3),
    "polystyrene": 1050/(1000**3),
    "wood": 897/(1000**3),
    "steel": 7930/(1000**3),
    'brick': 2200/(1000**3)

}
# Specific heat of the material in J/(kg*K) at 98 celcius
SpecificHeat = {
    "aluminium": 978,
    "pvc": 880,
    "glass": 840,
    "copper": 385,
    "polystyrene": 1300,
    "wood": 2380,
    "steel": 280,
    "brick": 800,
}
Absorptivity = {
    "aluminium": 1.0,
    "pvc": 1.0,
    "glass": 0.9,
    "copper": 0.9,
    "polystyrene": 0.9,
    "wood": 0.9,
    "steel": 0.9,
    "brick": 0.9,
}

TDiff = {
    "aluminium": 97.0, #97, #1.0, #97,
    "aluminium-6061": 64,
    "glass" : 0.34,
    "pvc": 0.17, # conductivity - 0.2 W/mK,
    "polystyrene": 0.5, #0.5,
    "copper": 111,
    "wood": 0.082,
    "steel": 4.2,
    "brick": 0.52,
}


TConductivity = {
    "aluminium": TDiff["aluminium"]*(Density["aluminium"]*SpecificHeat["aluminium"]),
    "pvc": TDiff["pvc"]*(Density["pvc"]*SpecificHeat["pvc"]),
    "glass": TDiff["glass"]*(Density["glass"]*SpecificHeat["glass"]),
    "copper": TDiff["copper"]*(Density["copper"]*SpecificHeat["copper"]),
    "polystyrene": TDiff["polystyrene"]*(Density["polystyrene"]*SpecificHeat["polystyrene"]),
    "wood": TDiff["wood"]*(Density["wood"]*SpecificHeat["wood"]),
    "steel": TDiff["steel"]*(Density["steel"]*SpecificHeat["steel"]),
    "brick": TDiff["brick"]*(Density["brick"]*SpecificHeat["brick"]),
}

# reflected energy
Emiss = EPSILON
distance = 0.4
TRefl = 22.22 #21.85

# atmospheric attenuation
TAtmC = 22.22 #21.85
TAtm = TAtmC + 273.15
Humidity = 30.0/100
Tau = 1.0

# external optics
TExtOptics = 20
# TransmissionExtOptics = 1.0
TransmissionExtOptics = 0.94

# Boson+ RBFO
# 6.23748836e+03 9.15243088e+01 1.08100666e+00 1.95966142e-01

# Boson RBFO
# R: 1080000.0, B: 1524.0, F: 1.0, O: 15500.0
R_const = {
    "Boson": 1080000.0,
    "Boson-Colocated": 2000000,
    "Boson+": 6.23748836e+03
}
B_const = {
    "Boson": 1524.0,
    "Boson-Colocated": 2137.05,
    "Boson+": 9.15243088e+01
}
F_const = {
    "Boson": 1.0,
    "Boson-Colocated": 219.46,
    "Boson+": 1.08100666e+00
}

O_const = {
    "Boson": 15500.0,
    "Boson-Colocated": 21470.7402,
    "Boson+": 1.95966142e-01
}

J0 = {
    "Boson": -1.07015226e+02,
    "Boson-Colocated": -1.07015226e+02,
    "Boson+": -1.07015226e+02
}
J1 = {
    "Boson": 3.73782461e-01,
    "Boson-Colocated": 3.73782461e-01,
    "Boson+": 3.73782461e-01
}

# Optimization parameters
DEFAULT_OPTIMIZATION_PARAMS: Dict[str, Any] = {
    'num_timesteps': 100,
    'dt': 0.1,
    'lr': 0.01,
    'max_iters': 1000,
    'tol': 1e-6
}

# Mesh parameters
DEFAULT_MESH_PARAMS: Dict[str, Any] = {
    'num_vertices': 1000,
    'num_faces': 2000,
    'scale': 1.0
}

def set_focus_distance(focus_distance: float) -> None:
    """
    Set camera focus distance.
    
    Args:
        focus_distance: New focus distance
    """
    global FOCUS_DISTANCE, IMAGE_WIDTH_METRIC, IMAGE_HEIGHT_METRIC
    FOCUS_DISTANCE = focus_distance
    print("FOCUS_DISTANCE set: ", FOCUS_DISTANCE)
    IMAGE_WIDTH_METRIC = 2*FOCUS_DISTANCE * \
        np.tan(np.deg2rad(HFOV/2))  # 158.58582759
    IMAGE_HEIGHT_METRIC = 2*FOCUS_DISTANCE * \
        np.tan(np.deg2rad(VFOV/2))  # 127.2686625

def set_FOV(hfov: float, vfov: float) -> None:
    """
    Set camera field of view.
    
    Args:
        hfov: Horizontal field of view in degrees
        vfov: Vertical field of view in degrees
    """
    global HFOV, VFOV, IMAGE_WIDTH_METRIC, IMAGE_HEIGHT_METRIC
    HFOV = hfov
    VFOV = vfov
    print("HFOV set: ", HFOV, "VFOV set: ", VFOV)
    IMAGE_WIDTH_METRIC = 2*FOCUS_DISTANCE * \
        np.tan(np.deg2rad(HFOV/2))  # 158.58582759
    IMAGE_HEIGHT_METRIC = 2*FOCUS_DISTANCE * \
        np.tan(np.deg2rad(VFOV/2))  # 127.2686625
    
def set_radiometry_constants(filename: str, linear: bool = True) -> None:
    """
    Set radiometry constants from file.
    
    Args:
        filename: Path to constants file
        linear: Whether to use linear interpolation
    """
    global P1, P2, R_const, B_const, F_const, O_const
    constants_dict = np.load(filename)
    if linear:

        m = float(constants_dict['m'])
        c = float(constants_dict['c'])
        P1 = 1.0/m
        P2 = -c/m
        print("Radiometry constants set: ", "P1: ", P1, "P2: ", P2)
    else:
        R_const[CAMERA] = float(constants_dict['R'])
        B_const[CAMERA] = float(constants_dict['B'])
        F_const[CAMERA] = float(constants_dict['F'])
        O_const[CAMERA] = float(constants_dict['O'])
        print("Radiometry constants set: ", "R: ", R_const[CAMERA], "B: ", B_const[CAMERA], "F: ", F_const[CAMERA], "O: ", O_const[CAMERA])

