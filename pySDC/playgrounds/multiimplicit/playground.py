# import pySDC.helpers.plot_helper as plt_helper
#
# import pickle
# import os
import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_3comp_mesh
from pySDC.implementations.sweeper_classes.multi_implicit import multi_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from pySDC.implementations.problem_classes.ReactionDiffusion_1D_FD_multiimplicit import reaction_diffusion_1d

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def main():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 500

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 0.01
    problem_params['nvars'] = 128
    problem_params['lambda0'] = 1.0
    problem_params['interval'] = (0, 1)
    problem_params['freq'] = 2

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['Q1'] = 'EE'
    sweeper_params['Q2'] = 'IE'
    sweeper_params['Q3'] = 'IE'
    # sweeper_params['spread'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = reaction_diffusion_1d
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_3comp_mesh
    description['sweeper_class'] = multi_implicit
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params

    # setup parameters "in time"
    t0 = 0
    Tend = 1.0
    dt_list = [Tend / 2 ** i for i in range(0, 4)]

    err = 0
    for dt in dt_list:
        print('Working with dt = %s...' % dt)

        level_params['dt'] = dt
        description['level_params'] = level_params

        # instantiate the controller
        controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                                   description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err_new = abs(uex - uend)

        print('error at time %s: %s' % (Tend, err_new))
        if err > 0:
            print('order of accuracy: %6.4f' % (np.log(err / err_new) / np.log(2)))

        err = err_new

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        # compute and print statistics
        niters = np.array([item[1] for item in iter_counts])
        out = '   Mean number of iterations: %4.2f' % np.mean(niters)
        # f.write(out + '\n')
        print(out)
        out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        # f.write(out + '\n')
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' % \
              (int(np.argmax(niters)), int(np.argmin(niters)))
        # f.write(out + '\n')
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % \
              (float(np.std(niters)), float(np.var(niters)))
        # f.write(out + '\n')
        # f.write(out + '\n')
        print(out)

if __name__ == "__main__":
    main()
