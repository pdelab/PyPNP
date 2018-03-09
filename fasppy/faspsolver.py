import ctypes
import numpy as np
import scipy.sparse as sp
import faspstructs as fpst
import dolfin as df


dirfasp = "/usr/local/faspsolver/lib/libfasp.so"
dirumfpack = "/usr/lib/x86_64-linux-gnu/libumfpack.so"
_umf = ctypes.CDLL(dirumfpack, mode=ctypes.RTLD_GLOBAL)
lfasp = ctypes.CDLL(dirfasp, mode=ctypes.RTLD_GLOBAL)


class solver:
    def __init__(self):
        self.st_dict = {'CG': 1,
                        'BiCGstab': 2,
                        'MinRes': 3,
                        'GMRes': 4,
                        'vGMRes': 5,
                        'vFGMRes': 6,
                        'GCG': 7,
                        'AMG Solver': 21,
                        'FMG Solver': 22,
                        'SuperLU': 31,
                        'UMFPACK': 32,
                        'MUMPS': 33}

        self.prec_dict = {'None': 0,
                          'Diag': 1,
                          'AMG': 2,
                          'FMG': 3,
                          'ILU': 4,
                          'Schwarz': 5}

        self.supported_prec = ['None', 'Diag', 'ILU']

        self.prec = 'None'
        self.itpar = fpst.itsolver_param()
        self.ilupar = fpst.ILU_param()
        self.amgpar = fpst.AMG_param()
        self.swzpar = fpst.SWZ_param()
        self.inpar = fpst.input_param()

    def read_params(self, input_file):
        cinput = ctypes.c_char_p(input_file)
        null_ptr = ctypes.POINTER(ctypes.c_int)()
        lfasp.fasp_param_input.restype = None
        lfasp.fasp_param_init.restype = None
        lfasp.fasp_param_input(cinput, ctypes.byref(self.inpar))

        lfasp.fasp_param_init(ctypes.byref(self.inpar),
                              ctypes.byref(self.itpar),
                              ctypes.byref(self.amgpar),
                              ctypes.byref(self.ilupar),
                              ctypes.byref(self.swzpar))

        prectype = self.itpar.prec()
        self.prec = [key for key, value in self.prec_dict.iteritems()
                     if value == prectype][0]

    def set_params(self, solvertype, prec, stop, itmax,
                   tol, restart, print_level):

        if solvertype not in self.st_dict.keys():
            print "Error: unknows solver type"
            exit()
        if prec not in self.supported_prec:
            print "Error: only preconditionner supported in set_params : ", \
                  self.supported_prec
            exit()

        self.prec = prec
        self.itpar = fpst.itsolver_param(self.st_dict[solvertype],
                                         self.prec_dict[prec], stop,
                                         itmax, tol, restart, print_level)

        self.ilupar = fpst.ILU_param(1, 2, 3, 0.01, 0.9, 0.01)

    def print_prec(self):
        print "The possible preconditionner are ", self.supported_prec

    def print_methods(self):
        print "The possible iteratives methods are ", self.st_dict.keys()

    def solve(self, A, b, x):
        row,col,val = df.as_backend_type(A).data()
        A_sp = sp.csr_matrix((val,col,row))
        b_v = np.array(b.get_local())
        x_v = np.array(x.vector().get_local())
        self.solve_csr(A_sp,b_v,x_v)

    def solve_csr(self, A, b, x):

        Afasp = fpst.dCSRmat(A.shape[0], A.shape[1], A.nnz,
                             np.ctypeslib.as_ctypes(A.indptr),
                             np.ctypeslib.as_ctypes(A.indices),
                             np.ctypeslib.as_ctypes(A.data))

        bfasp = fpst.dvector(3, np.ctypeslib.as_ctypes(b))
        xfasp = fpst.dvector(3, np.ctypeslib.as_ctypes(x))

        if self.prec == 'None':
            lfasp.fasp_solver_dcsr_krylov.restype = ctypes.c_int
            status = lfasp.fasp_solver_dcsr_krylov(
                                            ctypes.byref(Afasp),
                                            ctypes.byref(bfasp),
                                            ctypes.byref(xfasp),
                                            ctypes.byref(self.itpar))
        if self.prec == 'Diag':
            lfasp.fasp_solver_dcsr_krylov_diag.restype = ctypes.c_int
            status = lfasp.fasp_solver_dcsr_krylov_diag(
                                            ctypes.byref(Afasp),
                                            ctypes.byref(bfasp),
                                            ctypes.byref(xfasp),
                                            ctypes.byref(self.itpar))
        if self.prec == 'ILU':
            lfasp.fasp_solver_dcsr_krylov_ilu.restype = ctypes.c_int
            status = lfasp.fasp_solver_dcsr_krylov_ilu(
                                            ctypes.byref(Afasp),
                                            ctypes.byref(bfasp),
                                            ctypes.byref(xfasp),
                                            ctypes.byref(self.itpar),
                                            ctypes.byref(self.ilupar))
        if self.prec == 'AMG':
            lfasp.fasp_solver_dcsr_krylov_amg.restype = ctypes.c_int
            status = lfasp.fasp_solver_dcsr_krylov_amg(
                                            ctypes.byref(Afasp),
                                            ctypes.byref(bfasp),
                                            ctypes.byref(xfasp),
                                            ctypes.byref(self.itpar),
                                            ctypes.byref(self.amgpar))
        if self.prec == 'SWZ':
            lfasp.fasp_solver_dcsr_krylov_swz.restype = ctypes.c_int
            status = lfasp.fasp_solver_dcsr_krylov_swz(
                                            ctypes.byref(Afasp),
                                            ctypes.byref(bfasp),
                                            ctypes.byref(xfasp),
                                            ctypes.byref(self.itpar),
                                            ctypes.byref(self.swzpar))

        return status
