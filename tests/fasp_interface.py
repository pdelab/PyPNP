import ctypes
import numpy as np
import scipy.sparse as sp
import sys
from dolfin import *

sys.path.append('../')
import fasppy.faspsolver as fps


# Test with CSR Matrix
print "Test of FASP interface with CSR Matrix..."
Afull = np.array([ [1, 0, 0, 0, 0, 0],
                   [-1, 2, -1, 0, 0, 0],
                   [0, -1, 2, -1, 0, 0 ],
                   [0, 0, -1, 2, -1, 0 ],
                   [0, 0, 0, -1, 4, -1 ],
                   [0, 0, 0, 0, -1, 2]] , dtype=np.float64)
A = sp.csr_matrix(Afull)
b = np.array([1, 2, 3, 1, 1, 10],dtype=np.float64)
x = np.copy(b)
solver = fps.solver()
solver.set_params('GCG', 'ILU', 1, 100, 1e-6, 30, 0)
status = solver.solve_csr(A, b, x)
if np.linalg.norm(A.dot(x) - b ) < 1E-14:
    print "CSR MAtrix test is succesful"
else:
    print "CSR MAtrix test has an ERROR"

# Test With FENiCS
print "Test of FASP interface with FENiCS...",
if has_linear_algebra_backend("Eigen"):
        parameters["linear_algebra_backend"] = "Eigen"
else:
    print "DOLFIN has not been configured with Eigen."
    exit()
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds
x = Function(V)
x.interpolate(u0)
A = assemble(a)
b = assemble(L)
bc.apply(A,b)
solver2 = fps.solver()
solver2.read_params("./bsr.dat")
status = solver2.solve(A, b, x)
file = File("poisson.pvd")
file << x
# print np.linalg.norm(A.dot(x) - b )
