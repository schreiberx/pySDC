from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class reaction_diffusion_1d(ptype):
    """
    Example implementing the generalized Fisher's equation in 1D with finite differences

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'lambda0', 'interval', 'freq']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')
        if problem_params['freq'] >= 0 and problem_params['freq'] % 2 != 0:
            raise ProblemError('need even number of frequencies due to periodic BCs')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(reaction_diffusion_1d, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / self.params.nvars
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx)

    @staticmethod
    def __get_A(N, nu, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N))

        A = sp.diags(dstencil, doffsets, shape=(N, N), format='csc')
        A *= nu / (dx ** 2)

        return A

    # noinspection PyTypeChecker
    def solve_system_2(self, rhs, factor, u0, t):

        me = self.dtype_u(self.init)
        L = splu(sp.eye(self.params.nvars, format='csc') - factor * self.A)
        me.values = L.solve(rhs.values)
        return me

    def solve_system_3(self, rhs, factor, u0, t):

        me = self.dtype_u(self.init)
        me = 1.0 / (1.0 - factor * self.params.lambda0) * rhs
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        f = self.dtype_f(self.init)
        f.comp1 = self.__eval_comp1(u, t)
        f.comp2 = self.__eval_comp2(u, t)
        f.comp3 = self.__eval_comp3(u, t)
        return f

    def __eval_comp1(self, u, t):
        """
        Helper routine to evaluate the comp1 part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            dtype_f: comp1 part of RHS
        """

        comp1 = 0.0 * u

        return comp1

    def __eval_comp2(self, u, t):
        """
        Helper routine to evaluate the comp2 part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            dtype_f: comp2 part of RHS
        """

        comp2 = self.dtype_u(self.init)
        comp2.values = self.A.dot(u.values)
        return comp2

    def __eval_comp3(self, u, t):
        """
        Helper routine to evaluate the comp3 part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            dtype_f: comp3 part of RHS
        """

        comp3 = self.params.lambda0 * u
        return comp3

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([i * self.dx for i in range(self.params.nvars)])
        rho = (2 * np.cos(self.dx * self.params.freq * np.pi) - 2) / self.dx ** 2
        me.values = np.sin(np.pi * self.params.freq * xvalues) * \
            np.exp(t * self.params.nu * rho + t * self.params.lambda0)
        return me
