"""Small script to plot latin hypercubes. Separate so it works without X forwarding"""
import matplotlib.pyplot as plt
import numpy as np

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
