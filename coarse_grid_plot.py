"""Generate a test plot for an emulator"""
from __future__ import print_function
import os.path
import re
import numpy as np
import coarse_grid
import flux_power
import matter_power
import mean_flux as mflux
import scipy.spatial
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

def plot_convexhull(emulatordir):
    """Plot the convex hull of the projection of the emulator parameters"""
    params = coarse_grid.Emulator(emulatordir, mf=None)
    params.load()
    points = params.sample_params
    hull = scipy.spatial.ConvexHull(points)
    K = np.shape(points)[1]
    fig, axes = plt.subplots(K, K)
    for i in range(K):
        for j in range(K):
            ax = axes[i,j]
            if j >= i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.plot(points[:,i], points[:,j], 'o')
            projected = np.vstack([points[:,i], points[:,j]]).T
            hull = scipy.spatial.ConvexHull(projected)
            for simplex in hull.simplices:
                ax.plot(projected[simplex, 0], projected[simplex, 1], 'k-')
    return hull

def plot_test_interpolate(emulatordir,testdir, savedir=None, mean_flux=1, max_z=4.2, emuclass=None):
    """Make a plot showing the interpolation error."""
    if savedir is None:
        savedir = emulatordir
    myspec = flux_power.MySpectra(max_z=max_z)
    t0 = None
    if mean_flux:
        t0 = 0.95
    mf = mflux.ConstMeanFlux(value=t0)
    if mean_flux == 2:
        mf = mflux.MeanFluxFactor()
    params_test = coarse_grid.Emulator(testdir,mf=mf)
    params_test.load()
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf)
    else:
        params = emuclass(emulatordir, mf=mf)
    params.load()
    gp = params.get_emulator(max_z=max_z)
    kf = params.kf
    del params
    errlist = np.array([])
    #Constant mean flux.
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        if mean_flux == 2:
            pp = np.concatenate([[t0,], pp])
        predicted,std = gp.predict(pp.reshape(1,-1))
        ps = myspec.get_snapshot_list(dd)
        tfac = t0*mflux.obs_mean_tau(myspec.zout)
        exact = ps.get_power(kf = kf, tau0_factors = tfac)
        ratio = predicted[0]/exact
        upper = (predicted[0] + std[0])/exact
        lower = (predicted[0] - std[0])/exact
        errlist = np.concatenate([errlist, (predicted[0] - exact)/std[0]])
        #REMOVE
        plt.hist((predicted[0]-exact)/std[0],bins=100) #, density=True) #No 'density' property in Matplotlib v1
        xx = np.arange(-6, 6, 0.01)
        plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        plt.xlim(-6,6)
        plt.savefig(os.path.join(savedir, "errhist_"+str(np.size(errlist))+".pdf"))
        plt.clf()
        #DONE
        nred = len(myspec.zout)
        nk = len(kf)
        assert np.shape(ratio) == (nred*nk,)
        for i in range(nred):
            plt.semilogx(kf,ratio[i*nk:(i+1)*nk],label=myspec.zout[i])
            plt.fill_between(kf,lower[i*nk:(i+1)*nk], upper[i*nk:(i+1)*nk],alpha=0.3, color="grey")
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        name = params_test.build_dirname(pp, include_dense=True)
#         plt.title(name)
        plt.xlim(xmax=0.05)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
        if mean_flux:
            name = name+"mf0.95"
        name = re.sub(r"\.","_",str(name))+".pdf"
        #So we can use it in a latex document
        plt.savefig(os.path.join(savedir, name))
        print(name)
        plt.clf()
    #Plot the distribution of errors, compared to a Gaussian
    if np.all(np.isfinite(errlist)):
        plt.hist(errlist,bins=100) #, density=True)
        xx = np.arange(-6, 6, 0.01)
        plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        plt.xlim(-6,6)
        plt.savefig(os.path.join(savedir, "errhist.pdf"))
        plt.clf()
    return gp

def plot_test_matter_interpolate(emulatordir,testdir, redshift=3.):
    """Make a plot showing the interpolation error for the matter power spectrum."""
    params = coarse_grid.MatterPowerEmulator(emulatordir)
    params.load()
    gp = params.get_emulator()
    params_test = coarse_grid.MatterPowerEmulator(testdir)
    params_test.load()
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        predicted = gp.predict(pp)
        exact = matter_power.get_matter_power(dd,params.kf, redshift=redshift)
        ratio = predicted[0]/exact
        name = params_test.build_dirname(pp)
        plt.semilogx(params.kf,ratio,label=name)
    plt.xlabel(r"$k$ (h/kpc)")
    plt.ylabel(r"Predicted/Exact")
    plt.title("Matter power")
    plt.legend(loc=0)
    plt.show()
    plt.savefig(testdir+"matter_power.pdf")
    print(testdir+"matter_power.pdf")
    plt.clf()
    return gp
