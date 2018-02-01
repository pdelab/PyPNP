#! /usr/bin/python2.7
"""
    This a script to solve the PNP equations

    To run this programs:
        "python2 pnp.py params.py"
"""
from dolfin import *
import numpy as np
from pnpmodule import *

import sys
import imp
params = imp.load_source("params", sys.argv[1])
print "Important parameters from ", sys.argv[1]


print '##################################################'
print '#### Solving the PNP equations                ####'
print '##################################################'

# Chose the backend type
if has_linear_algebra_backend("PETSc"):
    parameters["linear_algebra_backend"] = "PETSc"
elif has_linear_algebra_backend("Eigen"):
    parameters["linear_algebra_backend"] = "Eigen"
else:
    print "DOLFIN has not been configured with PETSc or Eigen."
    exit()

parameters["allow_extrapolation"] = True

# Check and create the directories
DATA_DIR = params.DATA_DIR
IMG_DIR = params.IMG_DIR
files.CheckDir(DATA_DIR, params.CLEAN)
files.CheckDir(IMG_DIR, params.CLEAN)

# Create mesh and define function space
P1 = Point(-params.Lx/2.0, -params.Ly/2.0, -params.Lz/2.0)
P2 = Point(params.Lx/2.0, params.Ly/2.0, params.Lz/2.0)
mesh = BoxMesh(P1, P2, params.Nx, params.Ny, params.Nz)
FMesh = File(IMG_DIR+"mesh.pvd")    # Plot the Mesh
FMesh << mesh
DMesh = File(DATA_DIR+"mesh.xml")  # Print the Mesh
DMesh << mesh

# Two ways to do it Python or C++
coordinates = np.array(params.coordinates, dtype=np.uintp)
mesh_mins = np.array([-params.Lx/2.0, -params.Lx/2.0,
                      -params.Lx/2.0], dtype=np.float64)
mesh_maxs = np.array([params.Lx/2.0, params.Lx/2.0,
                     params.Lx/2.0], dtype=np.float64)
lower_values = np.array(params.lower_values, dtype=np.float64)
upper_values = np.array(params.upper_values, dtype=np.float64)
# Python
# uExpression = expressions.Linear_Functions(coordinates,
#                                            mesh_mins, mesh_maxs,
#                                            lower_values, upper_values,
#                                            degree=2)
# C++
uExpression = Expression(expressions.LinearFunctions_cpp, degree=2)
uExpression.update(coordinates, mesh_mins, mesh_maxs,
                   lower_values, upper_values)


def boundary(x, on_boundary):
    return ((x[0] < -params.Lx/2.0+5*DOLFIN_EPS
             or x[0] > params.Lx/2.0 - 5*DOLFIN_EPS)
            and on_boundary)


#  Finite Element Space
CG = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
CGFS = FunctionSpace(mesh, CG)
V = FunctionSpace(mesh, MixedElement((CG, CG, CG)))

u = TrialFunction(V)
v = TestFunction(V)

#  Previous Iterates
Solution = Function(V)
FFig = IMG_DIR+"Solution"
DData = DATA_DIR+"Solution"
uu = Function(V)
uu.interpolate(uExpression)

# Coefficients
eps = Constant(params.eps)
Dp = Constant(params.Dp)
qp = Constant(params.qp)
Dn = Constant(params.Dn)
qn = Constant(params.qn)

# Bilinear Form
a = (Dp*exp(uu[0]) * (inner(grad(u[0]), grad(v[0])) +
                      inner(grad(uu[0]) + qp * grad(uu[2]), grad(v[0]))
                      * u[0])) * dx \
    + (qp * Dp*exp(uu[0]) * inner(grad(u[2]), grad(v[0]))) * dx \
    + (Dn*exp(uu[1]) * (inner(grad(u[1]), grad(v[1])) +
                        inner(grad(uu[1]) + qn * grad(uu[2]), grad(v[1]))
                        * u[1])) * dx \
    + (qn*Dn*exp(uu[1]) * inner(grad(u[2]), grad(v[1]))) * dx \
    + (eps * inner(grad(u[2]), grad(v[2]))) * dx \
    + (-(qp*exp(uu[0])*u[0] + qn*exp(uu[1])*u[1])*v[2]) * dx

# Linear Form
L = - (Dp * exp(uu[0]) * (inner(grad(uu[0]), grad(v[0])) +
                          inner(grad(uu[0]) + qp*grad(uu[2]), grad(v[0]))
                          * uu[0]))*dx \
    - (qp * Dp * exp(uu[0]) * inner(grad(uu[2]), grad(v[0])))*dx \
    - (Dn*exp(uu[1]) * (inner(grad(uu[1]), grad(v[1])) + inner(grad(uu[1]) +
                        qn * grad(uu[2]), grad(v[1]))*uu[1]))*dx \
    - (qn * Dn * exp(uu[1]) * inner(grad(uu[2]), grad(v[1]))) * dx \
    - (eps * inner(grad(uu[2]), grad(v[2]))) * dx \
    - (- (qp * exp(uu[0])*uu[0] + qn * exp(uu[1]) * uu[1]) * v[2])*dx

# Parameters
tol = params.tol
itmax = params.itmax
it = 0
u0 = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, u0, boundary)
if parameters["linear_algebra_backend"] == "PETSc":
    solver = PETScKrylovSolver("gmres", params.linear_precon)
    solver.ksp().setGMRESRestart(params.gmres_restart)
if parameters["linear_algebra_backend"] == "Eigen":
    solver = EigenKrylovSolver("gmres", params.linear_precon)
solver.parameters["relative_tolerance"] = params.linear_tol
solver.parameters["maximum_iterations"] = params.linear_itmax
solver.parameters["nonzero_initial_guess"] = params.nonzero_initial_guess
solver.parameters["monitor_convergence"] = params.monitor_convergence


# Newton's Loop
print "Starting Newton's loop..."
nlsolvers.NewtonSolver(solver, a, L, V, bc, uu,
                       itmax, tol, FFig, DData,
                       Residual="relative", PrintFig=params.PrintFig,
                       PrintData=params.PrintData, Show=params.show)

print '##################################################'
print '#### End of the computation                   ####'
print '##################################################'
