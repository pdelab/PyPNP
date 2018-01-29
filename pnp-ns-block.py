#! /usr/bin/python2.7
"""
    This a script to solve the PNP equations
"""
from dolfin import *
import numpy as np
from LinearExp import *
from directories import *
from domains import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *


print '##################################################'
print '#### Solving the PNP equations                ####'
print '##################################################'

# Chose the backend type
parameters["linear_algebra_backend"] = "PETSc"
# parameters["linear_algebra_backend"] = "Eigen"
parameters["allow_extrapolation"] = True

# Check and create the directories
CLEAN = 'yes'
DATA_DIR = "DATA_PNP_NS/"
IMG_DIR = "IMG_PNP_NS//"
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
VitExpression= Expression(("0.0", "0.0","0.0"), degree=2)
PresExpression = Expression("0.0", degree=2)


def boundary(x, on_boundary):
    return ((x[0] < -Lx/2.0+5*DOLFIN_EPS
             or x[0] > Lx/2.0 - 5*DOLFIN_EPS)
            and on_boundary)

def boundary2(x, on_boundary):
    return on_boundary


#  Finite Element Space
CG = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
CGFS = FunctionSpace(mesh, CG)
V = FunctionSpace(mesh, MixedElement((CG, CG, CG)))

Vit  = FiniteElement("RT", mesh.ufl_cell(), 1)
Pres  = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
VitFS  = FunctionSpace(mesh, Vit)
PresFS  = FunctionSpace(mesh, Pres)
VNS = FunctionSpace(mesh, MixedElement((Vit,Pres)))

(Cat, An, Phi) = TrialFunction(V)
(cat, an, phi) = TestFunction(V)
(u, p) = TrialFunctions(VNS)
(v, q) = TestFunctions(VNS)

#  Previous Iterates
CatCat = Function(CGFS)
AnAn = Function(CGFS)
EsEs = Function(CGFS)
CatCat.interpolate(CationExpression)
AnAn.interpolate(AnionExpression)
EsEs.interpolate(PotentialExpression)
FCat = File(IMG_DIR+"Cat.pvd")
FAn = File(IMG_DIR+"An.pvd")
FPhi = File(IMG_DIR+"Phi.pvd")
uu = Function(VitFS)
pp = Function(PresFS)
uu.interpolate(VitExpression)
pp.interpolate(PresExpression)
FVit = File(IMG_DIR+"u.pvd")
FPres = File(IMG_DIR+"p.pvd")

# Coefficients
eps = Constant(1.0)
Dp = Constant(1.0)
qp = Constant(1.0)
Dn = Constant(1.0)
qn = Constant(-1.0)
f = Constant((0.0, 0.0, 0.0))
mu = Constant(1.0)
alpha1 = Constant(1.0)
alpha2 = Constant(1.0)

# Bilinear Form
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n_vec = FacetNormal(mesh)

a11 = (Dp*exp(CatCat) * (inner(grad(Cat), grad(cat)) +
                       inner(grad(CatCat) + qp * grad(EsEs), grad(cat))
                       * Cat)) * dx \
    + (qp * Dp*exp(CatCat) * inner(grad(Phi), grad(cat))) * dx \
    + (Dn*exp(AnAn) * (inner(grad(An), grad(an)) +
                       inner(grad(AnAn) + qn * grad(EsEs), grad(an))
                       * An)) * dx \
    + (qn*Dn*exp(AnAn) * inner(grad(Phi), grad(an))) * dx \
    + (eps * inner(grad(Phi), grad(phi))) * dx \
    + (-(qp*exp(CatCat)*Cat + qn*exp(AnAn)*An)*phi) * dx

a22  = ( 2.0*mu* inner( sym(grad(u)), sym(grad(v)) ) )*dx    -    ( p*div(v) )*dx   +   ( div(u)*q )*dx \
    + ( 2.0*mu*(alpha1)* inner( avg(sym(grad(u))), sym(outer(v('+'),n_vec('+')) + outer(v('-'),n_vec('-'))) ) )*dS \
    + ( 2.0*mu*(alpha1)* inner( sym(outer(u('+'),n_vec('+')) + outer(u('-'),n_vec('-'))), avg(sym(grad(v))) ) )*dS \
    + ( 2.0*mu*(alpha2/h_avg)* inner( jump(u),jump(v) ) )*dS \

a12  = - ( exp(CatCat)*(inner(u,grad(cat))) )*dx \
    - ( exp(AnAn)*(inner(u,grad(an)))  )*dx

a21  =  eps*inner( 2*outer(grad(Phi),grad(EsEs)) , grad(v) )*dx

# Linear Form
L1 = - (Dp * exp(CatCat) * (inner(grad(CatCat), grad(cat)) +
                           inner(grad(CatCat) + qp*grad(EsEs), grad(cat))
                           * CatCat))*dx \
    - (qp * Dp * exp(CatCat) * inner(grad(EsEs), grad(cat)))*dx \
    - (Dn*exp(AnAn) * (inner(grad(AnAn), grad(an)) + inner(grad(AnAn) +
                       qn * grad(EsEs), grad(an))*AnAn))*dx \
    - (qn * Dn * exp(AnAn) * inner(grad(EsEs), grad(an))) * dx \
    - (eps * inner(grad(EsEs), grad(phi))) * dx \
    - (- (qp * exp(CatCat)*CatCat + qn * exp(AnAn) * AnAn) * phi)*dx


L2  = - ( 2.0*mu* inner( sym(grad(uu)), sym(grad(v)) ) )*dx   +   ( pp*div(v) )*dx   -   ( div(uu)*q )*dx \
    - ( 2.0*mu*(alpha1)* inner( avg(sym(grad(uu))),  sym(outer(v('+'),n_vec('+')) + outer(v('-'),n_vec('-'))) ) )*dS \
    - ( 2.0*mu*(alpha1)* inner( sym(outer(uu('+'),n_vec('+')) + outer(uu('-'),n_vec('-'))), avg(sym(grad(v))) ) )*dS \
    - ( 2.0*mu*(alpha2/h_avg)* inner( jump(uu),jump(v) ) )*dS \
    - eps*inner( outer(grad(EsEs),grad(EsEs)) , grad(v) )*dx

# dofs
cat_dof = V.sub(0).dofmap().dofs()
an_dof = V.sub(1).dofmap().dofs()
phi_dof = V.sub(2).dofmap().dofs()
NV = len(cat_dof)+len(an_dof)+len(phi_dof)
vit_dof = VNS.sub(0).dofmap().dofs()
pres_dof = VNS.sub(1).dofmap().dofs()

# Parameters
tol = 1E-8
itmax = 20
it = 0
u0 = Constant((0.0, 0.0, 0.0))
bc0 = DirichletBC(V, u0, boundary)
bc1 = DirichletBC(VNS.sub(0), u0, boundary2)
domain  = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
domain.set_all(0)
domain.set_value(0, 1)
bc2 = DirichletBC(VNS.sub(1), 0.0, domain,1)

# Newton's Loop
print "Starting Newton's loop..."
bcs = [bc0, [bc1, bc2]]
AA = block_assemble([[a11, a12],
                     [a21,  a22 ]], bcs=bcs)
bb = block_assemble([L1, L2], bcs=bcs)
res = bb.norm("l2")
FCat << CatCat
FAn << AnAn
FPhi << EsEs
FVit << uu
FPres << pp

print "\t The initial residual is ", res
while (res > tol) and (it < itmax):
    # Extract the individual submatrices
    [[A, B],
     [C, D]] = AA
    # Ap = ILU(A)
    # Dp = AMG(D)
    # prec = block_mat([[Ap, 0],
    #                   [0, Dp]])
    prec = block_mat([[A, 0],
                      [C, D]])

    AAinv = LGMRES(AA, precond=prec, tolerance=1e-10, maxiter=500, show=2)
    sol_pnp, sol_ns = AAinv * bb

    CatCat.vector()[:] += sol_pnp[cat_dof]
    AnAn.vector()[:] += sol_pnp[an_dof]
    EsEs.vector()[:] += sol_pnp[phi_dof]

    uu.vector()[:] += sol_ns[vit_dof]
    pp.vector()[:] += sol_ns[pres_dof]

    FCat << CatCat
    FAn << AnAn
    FPhi << EsEs
    FVit << uu
    FPres << pp

    AA = block_assemble([[a11, a12],
                         [a21,  a22 ]], bcs=bcs)
    bb = block_assemble([L1, L2], bcs=bcs)
    res = bb.norm("l2")

    it += 1
    print "\t After ", it, " iterations the residual is ", res

print '##################################################'
print '#### End of the computation                   ####'
print '##################################################'
