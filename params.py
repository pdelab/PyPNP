#! /usr/bin/python2.7
"""
    Parameters
"""

# Folders
CLEAN = 'yes'
DATA_DIR = "output/data_pnp/"
IMG_DIR = "output/img_pnp/"

# Mesh
Nx = 25
Ny = 5
Nz = 5
Lx = 10.0
Ly = 2.0
Lz = 2.0

# Constants
eps = 1.0
Dp = 1.0
qp = 1.0
Dn = 1.0
qn = -1.0

# Initial values
coordinates = [0, 0, 0]
lower_values = [0, -2.0, -1.0]
upper_values = [-2.0, 0.0, 1.0]
