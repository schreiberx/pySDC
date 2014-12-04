from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.SWFW.ProblemClass import swfw_scalar
from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.Methods import sdc_step


if __name__ == "__main__":

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12

    sparams = {}
    sparams['Tend'] = 1
    sparams['maxiter'] = 100

    # This comes as read-in for the problem class
    cparams_f = {}
    cparams_f['lambda_s'] = -1j*0.1
    cparams_f['lambda_f'] = -1j*1.0
    cparams_f['u0'] = 1

    L0 = levclass.level(problem_class       =   swfw_scalar,
                        problem_params      =   cparams_f,
                        dtype_u             =   mesh,
                        dtype_f             =   rhs_imex_mesh,
                        collocation_class   =   collclass.CollGaussLegendre,
                        num_nodes           =   3,
                        sweeper_class       =   imex_1st_order,
                        level_params        =   lparams,
                        id                  =   'L0')


    S = stepclass.step(sparams)
    S.register_level(L0)


    S.time = 0.0
    S.dt = 1.0

    S.stats.niter = 0

    P = S.levels[0].prob
    uinit = P.u_exact(S.time)

    S.init_step(uinit)

    step_stats = []

    nsteps = int(S.params.Tend/S.dt)

    for n in range(nsteps):

        uend = sdc_step(S)

        step_stats.append(S.stats)

        S.time += S.dt

        S.reset_step()

        S.init_step(uend)


    uex = P.u_exact(S.time)

    # print(step_stats[1].residual,step_stats[1].level_stats[0].residual)

    print('error at time %s: %s' %(S.time,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)))
