#! /usr/bin/python2.7
"""
    Stokes Block solver
"""
from dolfin import *
import numpy as np
from pnpmodule import *
from scipy.sparse.linalg import gmres


"""
Stokes Matrix :
[ A B2 ] [ u ] = [ f ]
[ B 0  ] [ p ] = [ 0 ]
where B2 = B', A = laplacian, B = grad, B2 = -div
"""

A = np.ones(10)
B = np.ones(10)
B2 = np.ones(10)

def Astokes(u):
    zu = A.dot(u[vit_dof])+B2.dot(u[pres_dof])
    zp = B.dot(u[vit_dof])
    return np.array([zu, zp])

def PrecUzawa(u):
    AMG  = AMG(A)
    # Solve A*zu=u
    zu = gmres(A, u[vit_dof], tol=1e-05, restart=10, maxiter=1000, xtype=None, M=AMG, callback=None, restrt=10)
    zp = B.dot(u[vit_dof])-u[pres_dof]

LinStokes  =  LinearOperator((vit_dof+pres_dof,vit_dof+pres_dof), matvec=Astokes)
PrecStokes =  LinearOperator((vit_dof+pres_dof,vit_dof+pres_dof), matvec=PrecUzawa)

gmres(LinStokes, u[vit_dof], tol=1e-05, restart=10, maxiter=1000, xtype=None, M=PrecStokes callback=None, restrt=10)
