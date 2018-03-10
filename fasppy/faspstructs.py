import ctypes
import numpy as np
from scipy.sparse import csr_matrix

dirfasp = "/usr/local/faspsolver/lib/libfasp.so"
dirumfpack = "/usr/lib/x86_64-linux-gnu/libumfpack.so"
_umf = ctypes.CDLL(dirumfpack, mode=ctypes.RTLD_GLOBAL)
lfasp = ctypes.CDLL(dirfasp, mode=ctypes.RTLD_GLOBAL)


class dvector(ctypes.Structure):
    """
    /**
     * \struct dvector
     * \brief Vector with n entries of REAL type
     */
    typedef struct dvector{

        //! number of rows
        INT row;

        //! actual vector entries
        REAL *val;

    } dvector; /**< Vector of REAL type */
    """
    _fields_ = [("row", ctypes.c_int),
                ("val", ctypes.POINTER(ctypes.c_double))]


class dCSRmat(ctypes.Structure):
    """
     * struct dCSRmat
     * brief Sparse matrix of REAL type in CSR format
     *
     * CSR Format (IA,JA,A) in REAL
     *
     * note The starting index of A is 0.
     *
        //! row number of matrix A, m
        INT row;

        //! column of matrix A, n
        INT col;

        //! number of nonzero entries
        INT nnz;

        //! integer array of row pointers, the size is m+1
        INT *IA;

        //! integer array of column indexes, the size is nnz
        INT *JA;

        //! nonzero entries of A
        REAL *val;

    """
    _fields_ = [("row", ctypes.c_int),
                ("col", ctypes.c_int),
                ("nnz", ctypes.c_int),
                ("IA", ctypes.POINTER(ctypes.c_int)),
                ("JA", ctypes.POINTER(ctypes.c_int)),
                ("val", ctypes.POINTER(ctypes.c_double))]


class ILU_param(ctypes.Structure):
    """
    struct ILU_param

        //! print level
        SHORT print_level;

        //! ILU type for decomposition
        SHORT ILU_type;

        //! level of fill-in for ILUk
        INT ILU_lfil;

        //! drop tolerance for ILUt
        REAL ILU_droptol;

        //! add the sum of dropped elements to diagonal element
            in proportion relax
        REAL ILU_relax;

        //! permuted if permtol*|a(i,j)| > |a(i,i)|
        REAL ILU_permtol;

    Usually:
    ILU_type                 = 2      % 1 ILUk | 2 ILUt | 3 ILUtp
    ILU_lfil                 = 3      % level of fill-in for ILUk
    ILU_droptol              = 0.01   % ILU drop tolerance
    ILU_permtol              = 0.001  % permutation toleration for ILUtp
    ILU_relax                = 0.9    % add dropped entries to diagonal
                                        with relaxation
    """
    _fields_ = [("print_level", ctypes.c_short),
                ("ILU_type", ctypes.c_short),
                ("ILU_lfil", ctypes.c_int),
                ("ILU_droptol", ctypes.c_double),
                ("ILU_relax", ctypes.c_double),
                ("ILU_permtol", ctypes.c_double)]


class itsolver_param(ctypes.Structure):
    """
    struct itsolver_param

        SHORT itsolver_type; /**< solver type: see message.h */
        SHORT precond_type;  /**< preconditioner type: see message.h */
        SHORT stop_type;     /**< stopping criteria type */
        INT   maxit;         /**< max number of iterations */
        REAL  tol;           /**< convergence tolerance */
        INT   restart;       /**< number of steps for restarting:
                                    for GMRES etc */
        SHORT print_level;   /**< print level: 0--10 */


    solver_type   % 1 CG | 2 BiCGstab | 3 MinRes | 4 GMRes |
                  % 5 vGMRes | 6 vFGMRes | 7 GCG |
                  % 21 AMG Solver | 22 FMG Solver |
                  % 31 SuperLU | 32 UMFPACK | 33 MUMPS
    precond_type  % 0 None | 1 Diag | 2 AMG | 3 FMG | 4 ILU | 5 Schwarz
    stop_type     % 1 ||r||/||b|| | 2 ||r||_B/||b||_B | 3 ||r||/||x||
    """
    _fields_ = [("itsolver_type", ctypes.c_short),
                ("precond_type", ctypes.c_short),
                ("stop_type", ctypes.c_short),
                ("maxit", ctypes.c_int),
                ("tol", ctypes.c_double),
                ("restart", ctypes.c_int),
                ("print_level", ctypes.c_short)]

    def itsolver(self):
        return self.itsolver_type

    def prec(self):
        return self.precond_type


class AMG_param(ctypes.Structure):
    __fields__ = [("AMG_type", ctypes.c_short),
                  ("print_level", ctypes.c_short),
                  ("maxit", ctypes.c_int),
                  ("tol", ctypes.c_double),
                  ("max_levels", ctypes.c_short),
                  ("coarse_dof", ctypes.c_int),
                  ("cycle_type", ctypes.c_short),
                  ("quality_bound", ctypes.c_double),
                  ("smoother", ctypes.c_short),
                  ("smooth_order", ctypes.c_short),
                  ("presmooth_iter", ctypes.c_short),
                  ("postsmooth_iter", ctypes.c_short),
                  ("relaxation", ctypes.c_double),
                  ("polynomial_degree", ctypes.c_short),
                  ("coarse_solver", ctypes.c_short),
                  ("coarse_scaling", ctypes.c_short),
                  ("amli_degree", ctypes.c_short),
                  ("amli_coef", ctypes.POINTER(ctypes.c_double)),
                  ("nl_amli_krylov_type", ctypes.c_short),
                  ("coarsening_type", ctypes.c_short),
                  ("aggregation_type", ctypes.c_short),
                  ("interpolation_type", ctypes.c_short),
                  ("strong_threshold", ctypes.c_double),
                  ("max_row_sum", ctypes.c_double),
                  ("truncation_threshold", ctypes.c_double),
                  ("aggressive_level", ctypes.c_int),
                  ("aggressive_path", ctypes.c_int),
                  ("pair_number", ctypes.c_int),
                  ("strong_coupled", ctypes.c_double),
                  ("max_aggregation", ctypes.c_int),
                  ("tentative_smooth", ctypes.c_double),
                  ("smooth_filter", ctypes.c_short),
                  ("smooth_restriction", ctypes.c_short),
                  ("ILU_levels", ctypes.c_short),
                  ("ILU_type", ctypes.c_short),
                  ("ILU_lfil", ctypes.c_int),
                  ("ILU_droptol", ctypes.c_double),
                  ("ILU_relax", ctypes.c_double),
                  ("ILU_permtol", ctypes.c_double),
                  ("SWZ_levels", ctypes.c_int),
                  ("SWZ_mmsize", ctypes.c_int),
                  ("SWZ_maxlvl", ctypes.c_int),
                  ("SWZ_type", ctypes.c_int),
                  ("SWZ_blksolver", ctypes.c_int)]


class input_param(ctypes.Structure):
    __fields__ = [("print_level", ctypes.c_short),
                  ("output_type", ctypes.c_short),
                  ("inifile", ctypes.c_char*256),
                  ("workdir", ctypes.c_char*256),
                  ("problem_num", ctypes.c_int),
                  ("solver_type", ctypes.c_short),
                  ("precond_type", ctypes.c_short),
                  ("stop_type", ctypes.c_short),
                  ("itsolver_tol", ctypes.c_double),
                  ("itsolver_maxit", ctypes.c_int),
                  ("restart", ctypes.c_int),
                  ("ILU_type", ctypes.c_short),
                  ("ILU_lfil", ctypes.c_int),
                  ("ILU_droptol", ctypes.c_double),
                  ("ILU_relax", ctypes.c_double),
                  ("ILU_permtol", ctypes.c_double),
                  ("SWZ_mmsize", ctypes.c_int),
                  ("SWZ_maxlvl", ctypes.c_int),
                  ("SWZ_type", ctypes.c_int),
                  ("SWZ_blksolver", ctypes.c_int),
                  ("AMG_type", ctypes.c_short),
                  ("AMG_levels", ctypes.c_short),
                  ("AMG_cycle_type", ctypes.c_short),
                  ("AMG_smoother", ctypes.c_short),
                  ("AMG_smooth_order", ctypes.c_short),
                  ("AMG_relaxation", ctypes.c_double),
                  ("AMG_polynomial_degree", ctypes.c_short),
                  ("AMG_presmooth_iter", ctypes.c_short),
                  ("AMG_postsmooth_iter", ctypes.c_short),
                  ("AMG_coarse_dof", ctypes.c_int),
                  ("AMG_tol", ctypes.c_double),
                  ("AMG_maxit", ctypes.c_int),
                  ("AMG_ILU_levels", ctypes.c_short),
                  ("AMG_coarse_solver", ctypes.c_short),
                  ("AMG_coarse_scaling", ctypes.c_short),
                  ("AMG_amli_degree", ctypes.c_short),
                  ("AMG_nl_amli_krylov_type", ctypes.c_short),
                  ("AMG_SWZ_levels", ctypes.c_int),
                  ("AMG_coarsening_type", ctypes.c_short),
                  ("AMG_aggregation_type", ctypes.c_short),
                  ("AMG_interpolation_type", ctypes.c_short),
                  ("AMG_strong_threshold", ctypes.c_double),
                  ("AMG_truncation_threshold", ctypes.c_double),
                  ("AMG_max_row_sum", ctypes.c_double),
                  ("AMG_aggressive_level", ctypes.c_int),
                  ("AMG_aggressive_path", ctypes.c_int),
                  ("AMG_pair_number", ctypes.c_int),
                  ("AMG_quality_bound", ctypes.c_double),
                  ("AMG_strong_coupled", ctypes.c_double),
                  ("AMG_max_aggregation", ctypes.c_int),
                  ("AMG_tentative_smooth", ctypes.c_double),
                  ("AMG_smooth_filter", ctypes.c_short),
                  ("AMG_smooth_restriction", ctypes.c_short)]


class SWZ_param(ctypes.Structure):
    __fields__ = [("print_level", ctypes.c_short),
                  ("SWZ_type", ctypes.c_short),
                  ("SWZ_maxlvl", ctypes.c_int),
                  ("SWZ_mmsize", ctypes.c_int),
                  ("SWZ_blksolver", ctypes.c_int)]
