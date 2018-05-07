"""Make some plots"""
import sys

from make_paper_plots import *
from coarse_grid_plot import *
from plot_likelihood import *

if __name__ == "__main__":
    sim_rootdir = sys.argv[1]
    savedir = sys.argv[2]
    plotname = sys.argv[3]
    chain_savedir = sys.argv[4]

    testdir = sim_rootdir + '/emulator/hot_cold_test' #'/Lya_Boss/cosmo-only-test' #hires_s8_test' #/share/hypatia/sbird
    emudir = sim_rootdir + '/emulator/hot_cold' #'/Lya_Boss/cosmo-only-emulator' #hires_s8'

    likelihood_samples_plot_savefile = savedir + '/likelihood_samples_' + plotname + '.pdf'
    flux_power_plot_savefile = savedir + '/flux_power_' + plotname + '.pdf'

    test_knot_plots(testdir=testdir, emudir=emudir, plotdir=savedir, plotname=plotname, mf=1, kf_bin_nums=None, data_err=False, max_z=4.2)
    #plot_test_interpolate_kf_bin_loop(emudir, testdir, savedir=savedir, plotname="_Two_loop", kf_bin_nums=np.arange(2))

    #output = run_and_plot_likelihood_samples(testdir, emudir, likelihood_samples_plot_savefile, plotname, plot=True, chain_savedir=chain_savedir, n_burn_in_steps=5000, n_steps=15000, while_loop=False, mean_flux_label='s', return_class_only=False, include_emulator_error=True)
    #make_plot(chain_savedir + '/AA0.97BB1.3_chain_20000_MeanFluxFactor.txt', likelihood_samples_plot_savefile)
    #output = make_plot_flux_power_spectra(testdir, emudir, flux_power_plot_savefile, mean_flux_label='c')

    #plot_test_matter_interpolate(emudir, testdir, savedir=savedir)