#! /usr/bin/python2.7
"""
    This a script to solve the PNP equations
"""
from dolfin import *
import numpy as np
from pnpmodule import *


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
CLEAN = 'yes'
DATA_DIR = "DATA_PNP/"
IMG_DIR = "IMG_PNP/"
files.CheckDir(DATA_DIR, CLEAN)
files.CheckDir(IMG_DIR, CLEAN)

# Create mesh and define function space
Lx = 10.0
Ly = 2.0
Lz = 2.0
P1 = Point(-Lx/2.0, -Ly/2.0, -Lz/2.0)
P2 = Point(Lx/2.0, Ly/2.0, Lz/2.0)
mesh = BoxMesh(P1, P2, 25, 5, 5)
FMesh = File(IMG_DIR+"mesh.pvd")    # Plot the Mesh
FMesh << mesh
DMesh = File(DATA_DIR+"mesh.xml")  # Print the Mesh
DMesh << mesh

# Two ways to do it Python or C++
CationExpression = Expression(expressions.LinearFunction_cpp, degree=2)
CationExpression .update(0, -Lx/2.0, Lx/2.0, 0.0, -2.0)
AnionExpression = Expression(expressions.LinearFunction_cpp, degree=2)
AnionExpression.update(0, -Lx/2.0, Lx/2.0, -2.0, 0.0)
PotentialExpression = Expression(expressions.LinearFunction_cpp, degree=2)
PotentialExpression.update(0, -Lx/2.0, Lx/2.0, -1.0, 1.0)


def boundary(x, on_boundary):
    return ((x[0] < -Lx/2.0+5*DOLFIN_EPS
             or x[0] > Lx/2.0 - 5*DOLFIN_EPS)
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
eps = Constant(1.0)
Dp = Constant(1.0)
qp = Constant(1.0)
Dn = Constant(1.0)
qn = Constant(-1.0)

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
tol = 1E-8
itmax = 20
it = 0
u0 = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, u0, boundary)
if parameters["linear_algebra_backend"] == "PETSc":
    solver = PETScKrylovSolver("gmres", "ilu")
    solver.ksp().setGMRESRestart(10)
if parameters["linear_algebra_backend"] == "Eigen":
    solver = EigenKrylovSolver("gmres", "ilu")
solver.parameters["relative_tolerance"] = 1E-8
solver.parameters["maximum_iterations"] = 1000
solver.parameters["nonzero_initial_guess"] = True
solver.parameters["monitor_convergence"] = False


# Newton's Loop
print "Starting Newton's loop..."
nlsolvers.NewtonSolver(solver,a,L,V,[bc],[CatCat,AnAn,EsEs],
        itmax,tol,[FCat,FAn,FPhi],[DCat,DAn,DPhi],
        Residual="relative", PrintFig=1,PrintData=1,Show=2)

print '##################################################'
print '#### End of the computation                   ####'
print '##################################################'
