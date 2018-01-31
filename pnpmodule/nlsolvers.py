#! /usr/bin/python2.7
"""
    Newton Solver Loop
"""
from dolfin import *
import numpy as np


def NewtonSolver(solver, a, L, V, bcs, u, itmax, tol,
                 FigFiles, DFiles, Residual="true",
                 PrintFig=0, PrintData=0, Show=0):

    if (Residual is not "true") and (Residual is not "relative"):
        print "Wrong residual"
        exit()

    it = 0
    if isinstance(u, list) is False:
        u = [u]
    if isinstance(FigFiles, list)  is False:
        FigFiles = [FigFiles]
    if isinstance(DFiles, list)  is False:
        DFiles = [DFiles]
    Number_of_var = len(u)

    if (PrintData > 0) and (it % PrintData == 0):
        for i in range(Number_of_var):
            _DFiles = File(DFiles[i]+"_"+str(it)+".xml")
            _DFiles << u[i]

    _FigFiles = []
    if (PrintFig > 0) and (it % PrintFig == 0):
        for i in range(Number_of_var):
            _FigFiles.append(File(FigFiles[i]+".pvd"))
            _FigFiles[i] << u[i]

    Solution = Function(V)
    b = assemble(L)
    A = assemble(a)
    for bc in bcs:
        bc.apply(A, b)
    if Residual == "true":
        res = b.norm("l2")
    elif Residual == "relative":
        res = 1.0
        res_init = b.norm("l2")
    if (Show > 2):
        print "\t The initial true residual is ", res_init

    while (res > tol) and (it < itmax):
        solver.solve(A, Solution.vector(), b)
        if Number_of_var > 1:
            Temp = Solution.split(True)
        else:
            Temp = [Solution]
        for i in range(Number_of_var):
            u[i].vector()[:] += Temp[i].vector()[:]

        b = assemble(L)
        for bc in bcs:
            bc.apply(b)
        if Residual == "true":
            new_res = b.norm("l2")
        if Residual == "relative":
            new_res = b.norm("l2") / res_init
        lin_it = 0
        while (new_res > res) and (lin_it < 10):
            for i in range(Number_of_var):
                u[i].vector()[:] -= Temp[i].vector()[:] / (2.0**lin_it)
                u[i].vector()[:] += Temp[i].vector()[:] / (2.0**(lin_it+1))
            b = assemble(L)
            for bc in bcs:
                bc.apply(b)
            if Residual == "true":
                new_res = b.norm("l2")
            elif Residual == "relative":
                new_res = b.norm("l2") / res_init
            lin_it += 1
        res = new_res

        for i in range(Number_of_var):
            if (PrintFig > 0) and (it % PrintFig == 0):
                _FigFiles[i] << u[i]
            if (PrintData > 0) and (it % PrintData == 0):
                _DFiles = File(DFiles[i]+"_"+str(it)+".xml")
                _DFiles << u[i]

        A = assemble(a)
        bc.apply(A)
        it += 1
        if (Show == 2):
            print "\t After ", it, " iterations the ", Residual,\
                    " residual is ", res

    if (Show > 0):
        if (res < tol):
            print "\t Newton did converge in ", it, " iterations and the ",\
                    Residual, " residual is ", res
        else:
            print "Newton did not converge in ", it, " iterations and the ", \
                    Residual, "residual is ", res
