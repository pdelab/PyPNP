#! /usr/bin/python2.7
"""
domains
"""
from dolfin import *
import numpy as np


def apply_bc(A,b,bc,bc_values,dof_set):
    bc.apply(A,b)
    A.ident_local(dof_set)
    A.apply('insert')

    b_values = b.array()
    b_values[dof_set] = bc_values[np.arange(len(dof_set))]
    b.set_local(b_values)
    b.apply('insert')

def apply_bc_zero(A,bc,bc_values,dof_set):
    bc.zero(A)
    A.zero_local(dof_set)
    A.apply('insert')
