"""Generate a test plot for an emulator"""
from __future__ import print_function
import os.path
import re
import numpy as np
import coarse_grid
import flux_power
import matter_power
from mean_flux import ConstMeanFlux
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

def plot_test_interpolate(emulatordir,testdir, savedir=None, mean_flux=True, max_z=4.2, emuclass=None):
    """Make a plot showing the interpolation error."""
    if savedir is None:
        savedir = emulatordir
    t0 = None
    if mean_flux:
        t0 = 0.95
    mf = ConstMeanFlux(value=t0)
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf)
    else:
        params = emuclass(emulatordir, mf=mf)
    params.load()
    gp = params.get_emulator(max_z=max_z)
    kf = params.kf
    del params
    params_test = coarse_grid.Emulator(testdir)
    params_test.load()
    myspec = flux_power.MySpectra(max_z=max_z)
    #Constant mean flux.
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        predicted,std = gp.predict(pp.reshape(1,-1))
        ps = myspec.get_snapshot_list(dd)
        exact = ps.get_power(kf = kf, tau0_factors = t0)
        ratio = predicted[0]/exact
        upper = (predicted[0] + std[0])/exact
        lower = (predicted[0] - std[0])/exact
        nred = len(myspec.zout)
        nk = len(kf)
        assert np.shape(ratio) == (nred*nk,)
        for i in range(nred):
            plt.semilogx(kf,ratio[i*nk:(i+1)*nk],label=myspec.zout[i])
            plt.fill_between(kf,lower[i*nk:(i+1)*nk], upper[i*nk:(i+1)*nk],alpha=0.3, color="grey")
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        name = params_test.build_dirname(pp)
        plt.title(name)
        plt.xlim(xmax=0.05)
        plt.legend(loc=0)
        plt.show()
        if mean_flux:
            name = name+"mf"+str(t0)
        name = re.sub(r"\.","_",str(name))+".pdf"
        #So we can use it in a latex document
        plt.savefig(os.path.join(savedir, name))
        print(name)
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
