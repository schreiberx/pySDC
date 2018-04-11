import numpy as np
from pySDC.core.Sweeper import sweeper


class multi_implicit(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper using implicit/explicit Euler as base integrator

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'Q1' not in params:
            params['Q1'] = 'EE'
        if 'Q2' not in params:
            params['Q2'] = 'IE'
        if 'Q3' not in params:
            params['Q3'] = 'IE'

        # call parent's initialization routine
        super(multi_implicit, self).__init__(params)

        # Integration matrices
        self.Q1 = self.get_Qdelta_explicit(coll=self.coll, qd_type=self.params.Q1)
        self.Q2 = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.Q2)
        self.Q3 = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.Q3)

    def integrate(self):
        """
        Integrates the right-hand side (all three components)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].comp1 + L.f[j].comp2 + L.f[j].comp3)

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # subtract Q1F1(u^k)_m + Q2F2(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * (self.Q1[m + 1, j] * L.f[j].comp1 + self.Q2[m + 1, j] * L.f[j].comp2)
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau is not None:
                integral[m] += L.tau[m]

        Q3int = []
        for m in range(M):
            Q3int.append(P.dtype_u(P.init, val=0))
            for j in range(M + 1):
                Q3int[-1] += L.dt * self.Q3[m + 1, j] * L.f[j].comp3

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * (self.Q1[m + 1, j] * L.f[j].comp1 + self.Q2[m + 1, j] * L.f[j].comp2)

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system_2(rhs, L.dt * self.Q2[m + 1, m + 1], L.u[m + 1],
                                          L.time + L.dt * self.coll.nodes[m])
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])  # TODO: UGLY, remove

            rhs = L.u[m + 1] - Q3int[m]
            for j in range(m + 1):
                rhs += L.dt * self.Q3[m + 1, j] * L.f[j].comp3

            L.u[m + 1] = P.solve_system_3(rhs, L.dt * self.Q3[m + 1, m + 1], L.u[m + 1],  # TODO: is this a good guess?
                                          L.time + L.dt * self.coll.nodes[m])

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].comp1 + L.f[m + 1].comp2 + L.f[m + 1].comp3)
            # add up tau correction of the full interval (last entry)
            if L.tau is not None:
                L.uend += L.tau[-1]

        return None
