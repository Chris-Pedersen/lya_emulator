"""
This file contains functions which pick a set of samples from a parameter space
which will allow a Gaussian process to best interpolate the samples to new positions in parameter space.
Several schemes for this are possible.

We use rejection-sampled latin hypercubes.
"""

from scipy.stats.distributions import norm
import numpy as np


def _default_metric_func(lhs):
    """Default metric function for the maximinlhs, below.
    This is the sum of the Euclidean distances between each point and the closest other point."""
    #First find minimum distance between every two points
    nsamples, ndims = np.shape(lhs)
    #This is an array of the square of the distance between every two points, with dimensions (nsamp, nsamp)
    dists = np.array([np.sum((lhs - ll)**2,axis=1) for ll in lhs])
    assert np.shape(dists) == (nsamples, nsamples)
    #This is an array containing, for every point, the minimum distance to another point
    minn = np.array([np.min(dists[(i+1):,i]) for i in range(nsamples-1)])
    assert np.shape(minn) == (nsamples - 1,)
    assert np.all(minn > 0)
    return np.sqrt(np.sum(minn))

def maximinlhs(n, samples, prior_points = None, metric_func = None, maxlhs = 10000):
    """Generate multiple latin hypercubes and pick the one that maximises the metric function.
    Arguments:
    n: dimensionality of the cube to sample [0-1]^n
    samples: total number of samples.
    prior_points: List of previously evaluated points. If None, totally repopulate the space.
    metric_func: Function with which to judge the 'goodness' of the generated latin hypercube.
    Should be a scalar function of one hypercube sample set.
    maxlhs: Maximum number of latin hypercube to generate in total.
    Note convergence is pretty slow at the moment."""
    #Use the default metric if none is specified.
    if metric_func is None:
        metric_func = _default_metric_func
    #Minimal metric is zero.
    metric = 0
    group = 1000
    for _ in range(maxlhs//group):
        new = [lhscentered(n, samples, prior_points = prior_points) for _ in range(group)]
        new_metric = [metric_func(nn) for nn in new]
        best = np.argmax(new_metric)
        if new_metric[best] > metric:
            metric = new_metric[best]
            current = new[best]
    return current,metric

def remove_single_parameter(center, prior_points):
    """Remove all values within cells covered by prior samples for a particular parameter.
    Arguments:
    center contains the central values of each (evenly spaced) bin.
    prior_points contains the values of each already computed point."""
    #Find which bins the previously computed points are in
    already_taken = np.array([np.argmin(np.abs(center - pp)) for pp in prior_points])
    #Find the indices of points not in already_taken
    not_taken = np.setdiff1d(range(np.size(center)), already_taken)
    new_center = center[not_taken]
    assert np.size(new_center) == np.size(center) - np.size(prior_points)
    return new_center

def lhscentered(n, samples, prior_points = None):
    """
    Generate a latin hypercube design where all samples are
    centered on their respective cells. Can specify an already
    existing set of points using prior_points; these must also
    be a latin hypercube on a smaller sample, but need not be centered.
    """
    #Set up empty prior points if needed.
    if prior_points is None:
        prior_points = np.empty([0,n])

    npriors = np.shape(prior_points)[0]
    #Enforce that we are subsampling the earlier distribution, not supersampling it.
    assert samples > npriors
    new_samples = samples - npriors
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    # Number of stratified layers used is samples desired + prior_points.
    a = cut[:samples]
    b = cut[1:samples + 1]
    #Get list of central values
    _center = (a + b)/2
    # Choose a permutation so each sample is in one bin for each factor.
    H = np.zeros((new_samples, n))
    for j in range(n):
        #Remove all values within cells covered by prior samples for this parameter.
        #The prior samples must also be a latin hypercube!
        if npriors > 0:
            new_center = remove_single_parameter(_center, prior_points[:,j])
        else:
            new_center = _center
        H[:, j] = np.random.permutation(new_center)
    Hp = np.vstack((prior_points, H))
    assert np.shape(Hp) == (samples, n)
    return Hp

def map_from_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector from the unit cube to the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map. Should all be [0,1]
    param_limits - the maximal limits of the parameters to choose.
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all((0 <= param_vec)*(param_vec <= 1))
    assert np.all(param_limits[:,0] < param_limits[:,1])
    new_params = param_limits[:,0] + param_vec*(param_limits[:,1] - param_limits[:,0])
    assert np.all(new_params < param_limits[:,1])
    assert np.all(new_params > param_limits[:,0])
    return new_params

def map_to_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector to the unit cube from the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map.
    param_limits - the limits of the allowed parameters.
    Returns:
    vector of parameters, all in [0,1].
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all(param_vec < param_limits[:,1])
    assert np.all(param_vec > param_limits[:,0])
    assert np.all(param_limits[:,0] < param_limits[:,1])
    new_params = (param_vec-param_limits[:,0])/(param_limits[:,1] - param_limits[:,0])
    assert np.all((0 <= new_params)*(new_params <= 1))
    return new_params

def weight_cube(sample, means, sigmas):
    """
    Here we want to weight each dimension in the cube by its cumulative distribution function. So for parameter p in [p1, p2] we sample from p ~ CDF^-1(p1, p2)
    TODO: How to do this when we only approximately know the likelihood?

    """
    #This samples from the inverse CDF
    return norm(loc=means, scale=sigmas).ppf(sample)




#Wrap the plotting scripts in a try block so it succeeds on X-less clusters
try:
    import matplotlib.pyplot as plt


    def plot_points_hypercube(lhs_xval, lhs_yval, color="blue"):
        """Make a plot of the hypercube output points positioned on a regular grid"""
        ndivision = np.size(lhs_xval)
        assert ndivision == np.size(lhs_yval)
        xticks = np.linspace(0,1,ndivision+1)
        plt.scatter(lhs_xval, lhs_yval, marker='o', s=300, color=color)
        plt.grid(b=True, which='major')
        plt.xticks(xticks)
        plt.yticks(xticks)
        plt.xlim(0,1)
        plt.ylim(0,1)

except ImportError:
    pass
