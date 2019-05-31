#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Recovery module

"""


import utils

import numpy as np

from copy import deepcopy
from pyunlocbox import functions, solvers


## MAIN FUNCTIONS ##

def regress(graph, denoising_function, cost_function, 
            analysis_op_direct=None, analysis_op_adjoint=None, analysis_op_specnorm=1., 
            denoising_lip_const=0., **kwargs):
    r"""
    Regress on a noisy signal by minimizing the sum of cost and denoising functions.
    
    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph on whose vertices the sampled signal lives.
    denoising_function : :class:`pyunlocbox.functions.func`
        A convex function quantifying the error with respect to the sampled values. 
        It should have :func:`pyunlocbox.functions.func._eval` method implemented,
        as well as either the :func:`pyunlocbox.functions.func._grad` or the 
        :func:`pyunlocbox.functions.func._prox` methods.
    cost_function : :class:`pyunlocbox.functions.func`
        A convex cost function imbuing the search space with a complexity measure. 
        It should have :func:`pyunlocbox.functions.func._eval` method implemented,
        as well as either the :func:`pyunlocbox.functions.func._grad` or the 
        :func:`pyunlocbox.functions.func._prox` methods.
    analysis_op_direct : callable
        The analysis operator mapping graph signals to the space where the cost 
        function should be computed. If None (default), it is set to the identity.
    analysis_op_adjoint : callable
        The adjoint of `analysis_op_direct`. If None (default), it is set to the 
        identity.
    analysis_op_specnorm : float
        An estimate of the spectral norm of the analysis operator. Needed for 
        setting the step size in the optimization procedure. (default is 1.)
    denoising_lip_const : float
        An estimate of the Lipschitz constant of the gradient of the denoising 
        function, whenever it applies. Needed for setting the step size in the
        optimization procedure. (default is 0.)
    **kwargs :
        Additional solver parameters, such as maximum number of iterations
        (maxit), relative tolerance on the objective (rtol), and verbosity
        level (verbosity). See :func:`pyunlocbox.solvers.solve` for the full
        list of options.
    
    Returns
    -------
    ndarray of float or complex
        An array with first dimension equal to `graph.n_vertices` containing the 
        interpolated signal.
        
    Notes
    -----
    The minimization problem is solved using the primal-dual proximal splitting 
    procedure found in [Komodakis & Pesquet, 2015], Algorithm 6.
        
    """
    
    if analysis_op_direct is None:
        analysis_op_direct = lambda z: z

    if analysis_op_adjoint is None:
        analysis_op_adjoint = lambda z: z
    
    # Starting point for the iterative procedure
    initial_point = np.zeros((graph.n_vertices,))
    
    # Assemble the functions in the optimization program according to
    # whether the cost and denoising functions are differentiable or not
    if 'GRAD' in cost_function.cap(analysis_op_direct(initial_point)):
        g = functions.dummy()
        
        L = lambda z: np.zeros((graph.n_vertices,))
        Lt = lambda z: np.zeros((graph.n_vertices,))
        L_specnorm = 0.
        
        if 'GRAD' in denoising_function.cap(initial_point):
            f = functions.dummy()
            
            h = functions.func()
            h._eval = lambda z: cost_function.eval(analysis_op_direct(z)) + \
                                denoising_function.eval(z)
            h._grad = lambda z: analysis_op_adjoint(
                                    cost_function.grad(analysis_op_direct(z))) + \
                                denoising_function.grad(z)
            
            lip_const = analysis_op_specnorm ** 2 + denoising_lip_const
            
        else: 
            f = deepcopy(denoising_function)
            
            h = functions.func()
            h._eval = lambda z: cost_function.eval(analysis_op_direct(z))
            h._grad = lambda z: analysis_op_adjoint(
                                    cost_function.grad(analysis_op_direct(z)))
            
            lip_const = analysis_op_specnorm ** 2
            
    else:
        g = deepcopy(cost_function)
        g._eval = lambda z: cost_function.eval(analysis_op_direct(z))
        
        L = analysis_op_direct
        Lt = analysis_op_adjoint
        L_specnorm = analysis_op_specnorm
        
        if 'GRAD' in denoising_function.cap(initial_point):
            f = functions.dummy()
            
            h = deepcopy(denoising_function)
            
            lip_const = denoising_lip_const
            
            
        else: 
            f = deepcopy(denoising_function)
            
            h = functions.dummy()
            
            lip_const = 0.

       
    # Call the `pyunlocbox` solver
    step = 0.5 / (1. + L_specnorm + lip_const)
    solver = solvers.mlfbf(L=L, Lt=Lt, step=step)
    problem = solvers.solve([f, g, h], x0=initial_point, solver=solver, **kwargs)

    return problem['sol']


def interpolate(sampled_vertices, sampled_values, **kwargs):
    r"""
    Interpolate a subsampled signal by minimizing a cost function.
    
    Parameters
    ----------
    sampled_vertices : ndarray of int
        An array containing at most `graph.n_vertices` integer entries indicating
        which graph vertices were sampled.
    sampled_values : ndarray of float or complex
        An array containing the signal values at the sampled vertices indexed by 
        `sampled_coordinates`.
    
    Notes
    -----
    This is just the :func:`regress()` method with a denoising function set to the 
    convex indicator of the sampling set.
        
    """
    
    # Indicator function of the set satisfying the interpolation contraints
    # `z(sampled_vertices) = sampled_values`
    denoising_function = functions.func()
    denoising_function._eval = lambda z: 0
    denoising_function._prox = lambda z, T: utils.interpolation_projection(
                                                                z, 
                                                                sampled_vertices, 
                                                                sampled_values)
    
    return regress(denoising_function=denoising_function,
                   **kwargs)


## SPECIAL CASES ##

def tv_interpolation(graph, sampled_vertices, sampled_values, **kwargs):
    r"""
    Solve an interpolation problem via graph total variation minimization.
    
    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph on whose vertices the sampled signal lives.
    sampled_vertices : ndarray of int
        An array containing at most `graph.n_vertices` integer entries indicating
        which graph vertices were sampled.
    sampled_values : ndarray of float or complex
        An array containing the signal values at the sampled vertices indexed by 
        `sampled_coordinates`.

    Notes
    -----
    This is just the :func:`interpolate()` method with the cost function set to 
    :math:`z \mapsto \| \nabla_G z \|_1`.
    
    """
    
    # Set the analysis operator
    analysis_op_direct = lambda z: graph.grad(z) # Graph gradient (incidence transposed)
    analysis_op_adjoint = lambda z: graph.div(z) # Graph divergent (incidence matrix)
    graph.estimate_lmax()
    analysis_op_specnorm = np.sqrt(graph.lmax)
    
    # Set the cost function
    cost_function = functions.norm_l1()

    return interpolate(sampled_vertices=sampled_vertices, 
                       sampled_values=sampled_values,
                       graph=graph,
                       cost_function=cost_function, 
                       analysis_op_direct=analysis_op_direct, 
                       analysis_op_adjoint=analysis_op_adjoint, 
                       analysis_op_specnorm=analysis_op_specnorm,
                       **kwargs)


def tv_least_sq(graph, sampled_vertices, sampled_values, denoising_param=1., **kwargs):
    r"""
    Solve a regression problem via graph total variation and least squares minimization.

    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph on whose vertices the sampled signal lives.
    sampled_vertices : ndarray of int
        An array containing at most `graph.n_vertices` integer entries indicating
        which graph vertices were sampled.
    sampled_values : ndarray of float or complex
        An array containing the signal values at the sampled vertices indexed by 
        `sampled_coordinates`.
    denoising_param : float
        The regularization parameter multiplying the least squares functional. See 
        :math:`\rho` in the notes below.
        (default is `1.`)

    Notes
    -----
    This is just the :func:`regress()` method with cost function set to 
    :math:`z \mapsto \| \nabla_G z \|_1` and denoising function set to 
    :math:`z \mapsto \rho \| Az - y \|_2^2`, where :math:`\rho` is the denoising 
    parameter, and :math:`y` is the vector of sampled values.

    """
    
    # Set the analysis operator
    analysis_op_direct = lambda z: graph.grad(z) # Graph gradient (incidence transposed)
    analysis_op_adjoint = lambda z: graph.div(z) # Graph divergent (incidence matrix)
    graph.estimate_lmax()
    analysis_op_specnorm = np.sqrt(graph.lmax)
    
    # Set the cost function
    cost_function = functions.norm_l1()
    
    def A(z): # Down-sampling operator
        return utils.sampling_restriction(z, sampled_vertices)
    
    def At(y): # Up-sampling operator
        return utils.sampling_embedding(graph.n_vertices, y, sampled_vertices)
    
    # Set the denoising function
    denoising_function = functions.norm_l2(lambda_=denoising_param, y=sampled_values, 
                                           A=A, At=At)
    denoising_lip_const = denoising_param

    return regress(graph=graph, 
                   denoising_function=denoising_function,
                   cost_function=cost_function, 
                   analysis_op_direct=analysis_op_direct, 
                   analysis_op_adjoint=analysis_op_adjoint, 
                   analysis_op_specnorm=analysis_op_specnorm,
                   denoising_lip_const=denoising_lip_const,
                   **kwargs)


def dirichlet_form_interpolation(graph, sampled_vertices, sampled_values, **kwargs):
    r"""
    Solve an interpolation problem via Dirichlet form minimization.
    
    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph on whose vertices the sampled signal lives.
    sampled_vertices : ndarray of int
        An array containing at most `graph.n_vertices` integer entries indicating
        which graph vertices were sampled.
    sampled_values : ndarray of float or complex
        An array containing the signal values at the sampled vertices indexed by 
        `sampled_coordinates`.

    Notes
    -----
    This is just the :func:`interpolate()` method with the cost function set to 
    :math:`z \mapsto \| \nabla_G z \|_2^2`.
    
    """
    
    # Set the analysis operator
    analysis_op_direct = lambda z: graph.grad(z) # Graph gradient (incidence transposed)
    analysis_op_adjoint = lambda z: graph.div(z) # Graph divergent (incidence matrix)
    graph.estimate_lmax()
    analysis_op_specnorm = np.sqrt(graph.lmax)
    
    # Set the cost function
    cost_function = functions.norm_l2()

    return interpolate(sampled_vertices=sampled_vertices, 
                       sampled_values=sampled_values, 
                       graph=graph,
                       cost_function=cost_function, 
                       analysis_op_direct=analysis_op_direct, 
                       analysis_op_adjoint=analysis_op_adjoint, 
                       analysis_op_specnorm=analysis_op_specnorm,
                       **kwargs)


def dirichlet_form_least_sq(graph, sampled_vertices, sampled_values, denoising_param=1.,
                            **kwargs):
    r"""
    Solve a regression problem via Dirichlet form and least squares minimization.

    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
        The graph on whose vertices the sampled signal lives.
    sampled_vertices : ndarray of int
        An array containing at most `graph.n_vertices` integer entries indicating
        which graph vertices were sampled.
    sampled_values : ndarray of float or complex
        An array containing the signal values at the sampled vertices indexed by 
        `sampled_coordinates`.
    denoising_param : float
        The regularization parameter multiplying the least squares functional. See 
        :math:`\rho` in the notes below.
        (default is `1.`)

    Notes
    -----
    This is just the :func:`regress()` method with cost function set to 
    :math:`z \mapsto \| \nabla_G z \|_2^2` and denoising function set to 
    :math:`z \mapsto \rho \| Az - y \|_2^2`, where :math:`\rho` is the denoising 
    parameter, and :math:`y` is the vector of sampled values.

    """

    # Set the analysis operator
    analysis_op_direct = lambda z: graph.grad(z) # Graph gradient (incidence transposed)
    analysis_op_adjoint = lambda z: graph.div(z) # Graph divergent (incidence matrix)
    graph.estimate_lmax()
    analysis_op_specnorm = np.sqrt(graph.lmax)
    
    # Set the cost function
    cost_function = functions.norm_l2()
    
    def A(z): # Down-sampling operator
        return utils.sampling_restriction(z, sampled_vertices)
    
    def At(y): # Up-sampling operator
        return utils.sampling_embedding(graph.n_vertices, y, sampled_vertices)
    
    # Set the denoising function
    denoising_function = functions.norm_l2(lambda_=denoising_param, y=sampled_values,
                                           A=A, At=At)
    denoising_lip_const = denoising_param

    return regress(graph=graph, 
                   denoising_function=denoising_function,
                   cost_function=cost_function, 
                   analysis_op_direct=analysis_op_direct, 
                   analysis_op_adjoint=analysis_op_adjoint, 
                   analysis_op_specnorm=analysis_op_specnorm,
                   denoising_lip_const=denoising_lip_const,
                   **kwargs)
