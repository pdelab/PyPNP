#! /usr/bin/python2.7
"""
    This a script to solve the PNP equations
"""
from dolfin import *
import numpy as np
from LinearExp import *
from directories import *


print '##################################################'
print '#### Solving the PNP equations                ####'
print '##################################################'

# Chose the backend type
parameters["linear_algebra_backend"] = "PETSc"
# parameters["linear_algebra_backend"] = "Eigen"
parameters["allow_extrapolation"] = True

# Check and create the directories
CLEAN = 'yes'
DATA_DIR = "DATA/"
IMG_DIR = "IMG/"
CheckDir(DATA_DIR, CLEAN)
CheckDir(IMG_DIR, CLEAN)

# Create mesh and define function space
Lx = 10.0
Ly = 2.0
Lz = 2.0
P1 = Point(-Lx/2.0, -Ly/2.0, -Lz/2.0)
P2 = Point(Lx/2.0, Ly/2.0, Lz/2.0)
mesh = BoxMesh(P1, P2, 50, 5, 5)
FMesh = File(IMG_DIR+"mesh.pvd")    # Plot the Mesh
FMesh << mesh
FMeshX = File(DATA_DIR+"mesh.xml")  # Print the Mesh
FMeshX << mesh

# Two ways to do it Python or C++
CationExpression = Linear_Function(0, -Lx/2.0, Lx/2.0, 0.0, -2.0, degree=2)
CationExpression = Expression(LinearFunction_cpp, degree=2)
CationExpression .update(0, -Lx/2.0, Lx/2.0, 0.0, -2.0)
AnionExpression = Expression(LinearFunction_cpp, degree=2)
AnionExpression.update(0, -Lx/2.0, Lx/2.0, -2.0, 0.0)
PotentialExpression = Expression(LinearFunction_cpp, degree=2)
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
FCat = File(IMG_DIR+"Cat.pvd")
FAn = File(IMG_DIR+"An.pvd")
FPhi = File(IMG_DIR+"Phi.pvd")

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
solver = PETScKrylovSolver("gmres", "ilu")
solver.set_tolerances(1E-8, 1E-8, 100, 500)

# Newton's Loop
print "Starting Newton's loop..."
b = assemble(L)
A = assemble(a)
bc.apply(A, b)
res = b.norm("l2")
print "\t The initial residual is ", res
while (res > tol) and (it < itmax):
    solver.solve(A, Solution.vector(), b)
    Temp = Solution.split(True)
    CatCat.vector()[:] += Temp[0].vector()[:]
    AnAn.vector()[:] += Temp[1].vector()[:]
    EsEs.vector()[:] += Temp[2].vector()[:]
    FCat << CatCat
    FAn << AnAn
    FPhi << EsEs
    b = assemble(L)
    A = assemble(a)
    bc.apply(A, b)
    res = b.norm("l2")
    it += 1
    print "\t After ", it, " iterations the residual is ", res

print '##################################################'
print '#### End of the computation                   ####'
print '##################################################'
