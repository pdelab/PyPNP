#! /usr/bin/python2.7
"""
    This a script to solve the PNP equations

    To run this programs:
        "python2 pnp.py params.py"
"""
from dolfin import *
import numpy as np
from pnpmodule import *
import ctypes
import fasppy.faspsolver as fps

import sys
import imp
params = imp.load_source("params", sys.argv[1])
print "Important parameters from ", sys.argv[1]


print '##################################################'
print '#### Solving the PNP equations                ####'
print '##################################################'

# Chose the backend type
if params.linear_solver=='PETSc' and has_linear_algebra_backend("PETSc"):
    parameters["linear_algebra_backend"] = "PETSc"
    params.linear_precon = params.linear_precon.lower()
elif params.linear_solver=='Eigen' and has_linear_algebra_backend("Eigen"):
    parameters["linear_algebra_backend"] = "Eigen"
    params.linear_precon = params.linear_precon.lower()
elif params.linear_solver=='FASP' and has_linear_algebra_backend("Eigen"):
        parameters["linear_algebra_backend"] = "Eigen"
        params.linear_precon = params.linear_precon.upper()
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
Lenghts = np.array(params.Lenghts)
P1 = Point(-Lenghts/2.0)
P2 = Point(Lenghts/2.0)
mesh = BoxMesh(P1, P2, params.N[0], params.N[1], params.N[2])
FMesh = File(IMG_DIR+"mesh.pvd")    # Plot the Mesh
FMesh << mesh
DMesh = File(DATA_DIR+"mesh.xml")  # Print the Mesh
DMesh << mesh

# Two ways to do it Python or C++
coordinates = np.array(params.coordinates, dtype=np.uintp)
lower_values = np.array(params.lower_values, dtype=np.float64)
upper_values = np.array(params.upper_values, dtype=np.float64)
CationExpression = Expression(expressions.LinearFunction_cpp, degree=2)
CationExpression.update(coordinates[0], -Lenghts[coordinates[0]]/2.0,
                        Lenghts[coordinates[0]]/2.0,
                        lower_values[0], upper_values[0])
AnionExpression = Expression(expressions.LinearFunction_cpp, degree=2)
AnionExpression.update(coordinates[1], -Lenghts[coordinates[1]]/2.0,
                       Lenghts[coordinates[0]]/2.0,
                       lower_values[1], upper_values[1])
PotentialExpression = Expression(expressions.LinearFunction_cpp, degree=2)
PotentialExpression.update(coordinates[2], -Lenghts[coordinates[2]]/2.0,
                           Lenghts[coordinates[2]]/2.0,
                           lower_values[2], upper_values[2])


def boundary(x, on_boundary):
    return ((x[params.DirCoord] <
             - Lenghts[params.DirCoord]/2.0+5 * DOLFIN_EPS
            or x[params.DirCoord] >
             Lenghts[params.DirCoord]/2.0 - 5 * DOLFIN_EPS)
            and on_boundary)


#  Finite Element Space
CG = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
CGFS = FunctionSpace(mesh, CG)
V = FunctionSpace(mesh, MixedElement((CG, CG, CG)))

(Cat, An, Phi) = TrialFunction(V)
(cat, an, phi) = TestFunction(V)

#  Previous Iterates
CatCat = Function(CGFS)
AnAn = Function(CGFS)
EsEs = Function(CGFS)
CatCat.interpolate(CationExpression)
AnAn.interpolate(AnionExpression)
EsEs.interpolate(PotentialExpression)
Solution = Function(V)
FCat = IMG_DIR+"Cat"
FAn = IMG_DIR+"An"
FPhi = IMG_DIR+"Phi"
DCat = DATA_DIR+"Cat"
DAn = DATA_DIR+"An"
DPhi = DATA_DIR+"Phi"

# Coefficients
eps = Constant(params.eps)
Dp = Constant(params.Dp)
qp = Constant(params.qp)
Dn = Constant(params.Dn)
qn = Constant(params.qn)

# Bilinear Form
a = (Dp*exp(CatCat) * (inner(grad(Cat), grad(cat)) +
                       inner(grad(CatCat) + qp * grad(EsEs), grad(cat))
                       * Cat)) * dx \
    + (qp * Dp*exp(CatCat) * inner(grad(Phi), grad(cat))) * dx \
    + (Dn*exp(AnAn) * (inner(grad(An), grad(an)) +
                       inner(grad(AnAn) + qn * grad(EsEs), grad(an))
                       * An)) * dx \
    + (qn*Dn*exp(AnAn) * inner(grad(Phi), grad(an))) * dx \
    + (eps * inner(grad(Phi), grad(phi))) * dx \
    + (-(qp*exp(CatCat)*Cat + qn*exp(AnAn)*An)*phi) * dx

# Linear Form
L = - (Dp * exp(CatCat) * (inner(grad(CatCat), grad(cat)) +
                           inner(grad(CatCat) + qp*grad(EsEs), grad(cat))
                           * CatCat))*dx \
    - (qp * Dp * exp(CatCat) * inner(grad(EsEs), grad(cat)))*dx \
    - (Dn*exp(AnAn) * (inner(grad(AnAn), grad(an)) + inner(grad(AnAn) +
                       qn * grad(EsEs), grad(an))*AnAn))*dx \
    - (qn * Dn * exp(AnAn) * inner(grad(EsEs), grad(an))) * dx \
    - (eps * inner(grad(EsEs), grad(phi))) * dx \
    - (- (qp * exp(CatCat)*CatCat + qn * exp(AnAn) * AnAn) * phi)*dx

# Parameters
tol = params.tol
itmax = params.itmax
it = 0
u0 = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, u0, boundary)
if params.linear_solver == 'PETSc':
    solver = PETScKrylovSolver("gmres", params.linear_precon)
    solver.ksp().setGMRESRestart(params.gmres_restart)
    solver.parameters["relative_tolerance"] = params.linear_tol
    solver.parameters["maximum_iterations"] = params.linear_itmax
    solver.parameters["nonzero_initial_guess"] = params.nonzero_initial_guess
    solver.parameters["monitor_convergence"] = params.monitor_convergence

elif params.linear_solver == "Eigen":
    solver = EigenKrylovSolver("gmres", params.linear_precon)
    solver.parameters["relative_tolerance"] = params.linear_tol
    solver.parameters["maximum_iterations"] = params.linear_itmax
    solver.parameters["nonzero_initial_guess"] = params.nonzero_initial_guess
    solver.parameters["monitor_convergence"] = params.monitor_convergence

elif params.linear_solver == "FASP":
    solver = EigenKrylovSolver("gmres", params.linear_precon)
    if  params.linear_solver_bsr is not None:
        solver.read_params( params.linear_solver_bsr)
    else:
        solver.set_params('GMRes',  params.linear_precon, 1,
                          params.linear_itmax, params.linear_tol,
                          params.gmres_restar, params.prints)

# Newton's Loop
print "Starting Newton's loop..."
nlsolvers.NewtonSolver(solver, a, L, V, [bc], [CatCat, AnAn, EsEs],
                       itmax, tol, [FCat, FAn, FPhi], [DCat, DAn, DPhi],
                       Residual="relative", PrintFig=params.PrintFig,
                       PrintData=params.PrintData, Show=params.show)

print '##################################################'
print '#### End of the computation                   ####'
print '##################################################'
