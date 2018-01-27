#! /usr/bin/python2.7
"""
    This a script to solve the PNP equations
"""
from dolfin import *
import numpy as np
from LinearExp import *
from directories import *
from domains import *

print '##################################################'
print '#### Solving the NS equations                 ####'
print '##################################################'

# Chose the backend type
parameters["linear_algebra_backend"] = "PETSc"
# parameters["linear_algebra_backend"] = "Eigen"
parameters["allow_extrapolation"] = True

# Check and create the directories
CLEAN = 'yes'
DATA_DIR = "DATA_NS/"
IMG_DIR = "IMG_NS/"
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
VitExpression= Expression(("-sin(x[0]*pi)", "0.0","0.0"), degree=2)
PresExpression = Expression("0.0", degree=2)

def boundary(x, on_boundary):
    return on_boundary


#  Finite Element Space
Vit  = FiniteElement("RT", mesh.ufl_cell(), 1)
Pres  = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
VitFS  = FunctionSpace(mesh, Vit)
PresFS  = FunctionSpace(mesh, Pres)
VNS = FunctionSpace(mesh, MixedElement((Vit,Pres)))

(u, p) = TrialFunctions(VNS)
(v, q) = TestFunctions(VNS)

#  Solutions
uu = Function(VitFS)
pp = Function(PresFS)
Solution = Function(VNS)
uu.interpolate(VitExpression)
pp.interpolate(PresExpression)
FVit = File(IMG_DIR+"u.pvd")
FPres = File(IMG_DIR+"p.pvd")

# Coefficients
f = Constant((0.0, 0.0, 0.0))
mu = Constant(1.0)
alpha1 = Constant(1.0)
alpha2 = Constant(1.0)

# Bilinear Form
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n_vec = FacetNormal(mesh)

a   = ( 2.0*mu* inner( sym(grad(u)), sym(grad(v)) ) )*dx    -    ( p*div(v) )*dx   +   ( div(u)*q )*dx \
	+ ( 2.0*mu*(alpha1)* inner( avg(sym(grad(u))), sym(outer(v('+'),n_vec('+')) + outer(v('-'),n_vec('-'))) ) )*dS \
	+ ( 2.0*mu*(alpha1)* inner( sym(outer(u('+'),n_vec('+')) + outer(u('-'),n_vec('-'))), avg(sym(grad(v))) ) )*dS \
	+ ( 2.0*mu*(alpha2/h_avg)* inner( jump(u),jump(v) ) )*dS


L   = inner(f,v)*dx

# Form for use in constructing preconditioner matrix
ap = ( 2.0*mu* inner( sym(grad(u)), sym(grad(v)) ) )*dx    + p*q*dx \
	+ ( 2.0*mu*(alpha1)* inner( avg(sym(grad(u))), sym(outer(v('+'),n_vec('+')) + outer(v('-'),n_vec('-'))) ) )*dS \
	+ ( 2.0*mu*(alpha1)* inner( sym(outer(u('+'),n_vec('+')) + outer(u('-'),n_vec('-'))), avg(sym(grad(v))) ) )*dS \
	+ ( 2.0*mu*(alpha2/h_avg)* inner( jump(u),jump(v) ) )*dS

# Parameters
tol = 1E-8
itmax = 20
it = 0
u0 = Constant((2.0, 0.0, 0.0))
bc = DirichletBC(VNS.sub(0), u0, boundary)
dof_set=np.array([0], dtype='intc')
bc_values = np.zeros(len(dof_set))


solver = PETScKrylovSolver("gmres", "amg")
solver.set_tolerances(1E-6, 1E-6, 100, 1000)
solver.parameters["monitor_convergence"] = True

# Newton's Loop
print "Starting Newton's loop..."
A = assemble(a)
b = assemble(L)
P = assemble(ap)
apply_bc(A,b,bc,bc_values,dof_set)
apply_bc(P,b,bc,bc_values,dof_set)

# solver.set_operator(A)
solver.set_operators(A, P)
solver.solve(Solution.vector(), b)
Temp = Solution.split(True)
uu.vector()[:] = Temp[0].vector()[:]
pp.vector()[:] = Temp[1].vector()[:]
FVit << uu
FPres << pp

print '##################################################'
print '#### End of the computation                   ####'
print '##################################################'
