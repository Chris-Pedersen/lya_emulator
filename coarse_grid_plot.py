"""Generate a test plot for an emulator"""
from __future__ import print_function
import os.path
import re
import numpy as np
import coarse_grid
import flux_power
import matter_power
import mean_flux as mflux
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

def plot_test_interpolate_kf_bin_loop(emulatordir, testdir, savedir=None, plotname="", kf_bin_nums=np.arange(1)):
    if savedir is None:
        savedir = emulatordir

    all_power_array_all_kf = [None] * kf_bin_nums.size
    for i in range(kf_bin_nums.size):
        plotname_single_kf_bin = plotname + '_' + str(kf_bin_nums[i])
        gp, all_power_array_all_kf[i], z_labs = plot_test_interpolate(emulatordir, testdir, savedir=savedir, plotname=plotname_single_kf_bin, kf_bin_nums=[kf_bin_nums[i],])

    all_power_array_all_kf = np.array(all_power_array_all_kf)
    for j in range(all_power_array_all_kf.shape[1]): #Loop over validation points in parameter space
        print("Validation point", j+1, "/", all_power_array_all_kf.shape[1])
        #Plot error histogram
        power_difference = all_power_array_all_kf[:, j, 1, :] - all_power_array_all_kf[:, j, 3, :]
        err_norm = power_difference / all_power_array_all_kf[:, j, 2, :]
        _plot_error_histogram(savedir, "_validation_parameters_" + str(j) + plotname, err_norm.flatten())

        #Plot predicted/exact
        power_ratio = all_power_array_all_kf[:, j, 1, :] / all_power_array_all_kf[:, j, 3, :]
        power_lower = (all_power_array_all_kf[:, j, 1, :] - all_power_array_all_kf[:, j, 2, :]) / all_power_array_all_kf[:, j, 3, :]
        power_upper = (all_power_array_all_kf[:, j, 1, :] + all_power_array_all_kf[:, j, 2, :]) / all_power_array_all_kf[:, j, 3, :]
        for k in range(all_power_array_all_kf.shape[3]): #Loop over redshift bins
            kf = all_power_array_all_kf[:, j, 0, k]
            plt.semilogx(kf, power_ratio[:, k], label=z_labs[k])
            plt.fill_between(kf, power_lower[:, k], power_upper[:, k], alpha=0.3, color="grey")
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        plt.xlim(xmax=0.05)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
        name = "validation_parameters_" + str(j) + plotname + ".pdf"
        plt.savefig(os.path.join(savedir, name))
        print(name)
        plt.clf()

    #Plot combined error histogram
    power_difference = all_power_array_all_kf[:, :, 1, :] - all_power_array_all_kf[:, :, 3, :]
    err_norm = power_difference / all_power_array_all_kf[:, :, 2, :]
    _plot_error_histogram(savedir, plotname, err_norm.flatten())

    #Save combined output
    array_savename = os.path.join(savedir, "combined_output" + plotname + '.npy')
    np.save(array_savename, all_power_array_all_kf)

def _plot_error_histogram(savedir, plotname, err_norm):
    plt.hist(err_norm, bins=100, density=True)
    xx = np.arange(-6, 6, 0.01)
    _plot_unit_Gaussians(xx)
    plt.xlim(-6, 6)
    plt.savefig(os.path.join(savedir, "errhist" + plotname + ".pdf"))
    plt.clf()

def _plot_unit_Gaussians(xx):
    plt.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ls="-", color="black")
    plt.plot(xx, np.exp(-xx ** 2 / 2 / 2 ** 2) / np.sqrt(2 * np.pi * 2 ** 2), ls="--", color="grey")

def plot_test_interpolate(emulatordir,testdir, savedir=None, plotname="", mean_flux=1, max_z=4.2, emuclass=None, kf_bin_nums=None):
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
    params_test = coarse_grid.Emulator(testdir,mf=mf, kf_bin_nums=kf_bin_nums)
    params_test.load()
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf, kf_bin_nums=kf_bin_nums)
    else:
        params = emuclass(emulatordir, mf=mf, kf_bin_nums=kf_bin_nums)
    params.load()
    gp = params.get_emulator(max_z=max_z)
    kf = params.kf
    del params
    errlist = np.array([])
    #Constant mean flux.

    # Save output
    nred = len(myspec.zout)
    nkf = kf.size
    #print("Number of validation points =", params_test.get_parameters().shape[0])
    all_power_array = np.zeros((params_test.get_parameters().shape[0], 4, nkf*nred)) #kf, predicted, std, exact
    validation_number = 0

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
        plt.hist((predicted[0]-exact)/std[0],bins=100 , density=True) #No 'density' property in Matplotlib v1
        xx = np.arange(-6, 6, 0.01)
        plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        plt.xlim(-6,6)
        plt.savefig(os.path.join(savedir, "errhist_"+str(np.size(errlist))+plotname+".pdf"))
        plt.clf()
        #DONE
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
        name = re.sub(r"\.","_",str(name))+plotname+".pdf"
        #So we can use it in a latex document
        plt.savefig(os.path.join(savedir, name))
        print(name)
        plt.clf()

        #Save output
        all_power_array[validation_number] = np.vstack((np.tile(kf, nred), predicted[0], std[0], exact))
        array_savename = os.path.join(savedir, name[:-4] + '.npy')
        np.save(array_savename, all_power_array[validation_number])
        validation_number+=1

    #Plot the distribution of errors, compared to a Gaussian
    if np.all(np.isfinite(errlist)):
        plt.hist(errlist,bins=100, density=True)
        xx = np.arange(-6, 6, 0.01)
        plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        plt.xlim(-6,6)
        plt.savefig(os.path.join(savedir, "errhist"+plotname+".pdf"))
        plt.clf()
    return gp, all_power_array, myspec.zout

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
