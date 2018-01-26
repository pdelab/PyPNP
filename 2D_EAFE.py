''' 
EAFE scheme for the H1 convection diffusion equation (in divergence
form): 

-div(diff_coef*grad u + beta u) = f
'''

from __future__ import division
from dolfin import *
import numpy as np
import sys

##################################################
# Parameters & convection function  
##################################################
# beta = (y, -x)
def convection(x) :
  return np.array([x[1],-x[0]])

# usage: diffusion coefficient N_mesh 
# Default parameters
diff_coef = 0.01 
N = 8

if len(sys.argv) > 1 :
  diff_coef = float(sys.argv[1])

if len(sys.argv) > 2 :
  N = int(sys.argv[2])

print "diffusion coefficient =", diff_coef 
print "N mesh =", N

# exact solution 
u_exact = Expression('sin(2*DOLFIN_PI*x[0])* \
                      cos(2*DOLFIN_PI*x[1])', degree=4)
f = Expression("8*DOLFIN_PI*DOLFIN_PI*D*sin(2*DOLFIN_PI*x[0])* \
                cos(2*DOLFIN_PI*x[1]) \
               -2*DOLFIN_PI*x[1]*cos(2*DOLFIN_PI*x[0])*cos(2*DOLFIN_PI*x[1]) \
               -2*DOLFIN_PI*x[0]*sin(2*DOLFIN_PI*x[0])*sin(2*DOLFIN_PI*x[1]) \
                ", degree=2, D=diff_coef)

##################################################
# 1D Bernoulli function 0-dim / 1-dim 
##################################################
def bernoulli1(r, diff) :
  eps = 1e-10
  if np.absolute(r)<diff*eps : 
    return diff 
  elif r < -diff*eps : # r < 0
    return r / np.expm1(r/diff) 
  else : # r > 0 
    return r*np.exp(-r/diff) / (1 - np.exp(-r/diff))

##################################################
# Mesh and FEM space
##################################################
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define the variational form
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

################################################## 
# Boundary condition
################################################## 
def boundary(x, on_boundary):
  return on_boundary 

#u0 = Constant(0.0)
bc = DirichletBC(V, u_exact, boundary)

################################################## 
# Build the stiffness matrix 
##################################################
A = assemble(a)
A.zero()
dof_map = V.dofmap() 
dof_coord = V.tabulate_dof_coordinates()

for cell in cells(mesh) :
  local_to_global_map = dof_map.cell_dofs(cell.index()) 
  # build the local tensor 
  local_tensor = assemble_local(a, cell)
  # EAFE: change the local tensor 
  # Step 1: Find the point related to dofs 
  a0 = np.array([ dof_coord[2*local_to_global_map[0]],
                  dof_coord[2*local_to_global_map[0]+1] ])
  a1 = np.array([ dof_coord[2*local_to_global_map[1]],
                  dof_coord[2*local_to_global_map[1]+1] ])
  a2 = np.array([ dof_coord[2*local_to_global_map[2]],
                  dof_coord[2*local_to_global_map[2]+1] ])
  bary_center = (a0+a1+a2) / 3
  # Step 2: Find the convection by local constant approximation
  beta = convection(bary_center)
  # Step 3: Apply bernoulli function 
  b01 = bernoulli1( np.inner(beta, a0-a1), diff_coef ) 
  b10 = bernoulli1( np.inner(beta, a1-a0), diff_coef )
  b02 = bernoulli1( np.inner(beta, a0-a2), diff_coef ) 
  b20 = bernoulli1( np.inner(beta, a2-a0), diff_coef )
  b12 = bernoulli1( np.inner(beta, a1-a2), diff_coef ) 
  b21 = bernoulli1( np.inner(beta, a2-a1), diff_coef ) 
  # Step 4: Change the local tensor 
  local_tensor[0][1] *= b01
  local_tensor[1][0] *= b10
  local_tensor[0][2] *= b02
  local_tensor[2][0] *= b20
  local_tensor[1][2] *= b12
  local_tensor[2][1] *= b21
  local_tensor[0][0] = -local_tensor[1][0] - local_tensor[2][0]
  local_tensor[1][1] = -local_tensor[0][1] - local_tensor[2][1]
  local_tensor[2][2] = -local_tensor[0][2] - local_tensor[1][2]
  # Build the stiffness matrix 
  A.add(local_tensor, local_to_global_map, local_to_global_map)
  A.apply('insert')

# Build the right-hand-side
b = assemble(L)
# Apply boundary condition
bc.apply(A, b)

##################################################
# Compute solution
##################################################
u = Function(V)

# Solver the linear system 
solver = LUSolver(A,"default");
solver.parameters["symmetric"] = False;
solver.solve(u.vector(), b)

################################################## 
# Error and plot 
##################################################
print "N =", N, ": L2 error =", errornorm(u_exact, u, 'l2', 3)
print "N =", N, ": H1 error =", errornorm(u_exact, u, 'H1', 3)

### Save solution in VTK format
##file = File("2DH1-dual-EAFE.pvd")
##file << u

# end of file #
