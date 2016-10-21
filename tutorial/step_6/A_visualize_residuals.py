import numpy as np
import os

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats
from pySDC.plugins.visualization_tools import show_residual_across_simulation


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 5E-10
    level_params['dt'] = 0.125

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1           # diffusion coefficient
    problem_params['freq'] = 2           # frequency for the test value
    problem_params['nvars'] = [63,31]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d                           # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = mesh                                   # pass data type for f
    description['sweeper_class'] = generic_LU                       # pass sweeper
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh              # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # set up list of parallel time-steps to run PFASST with
    num_proc = 8

    # instantiate controller
    controller = allinclusive_classic_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)
    print('Error: %12.8e' % (err))

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    min_iter = 99
    max_iter = 0
    for item in iter_counts:
        print('Number of iterations for time %4.2f: %1i' % item)
        min_iter = min(min_iter,item[1])
        max_iter = max(max_iter,item[1])

    show_residual_across_simulation(stats,'residuals.png')

    assert os.path.isfile('residuals.png')
    assert min_iter == 5 and max_iter == 7, "ERROR: number of iterations not as expected, got %s and %s" %(min_iter, max_iter)

if __name__ == "__main__":
    main()
