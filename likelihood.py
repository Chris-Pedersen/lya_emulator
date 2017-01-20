"""Module for computing the likelihood function for the forest emulator."""
import os
import os.path
import numpy as np
import emcee
import coarse_grid
import flux_power
import gpemulator


def _siIIIcorr(kf):
    """For precomputing the shape of the SiIII correlation"""
    #Compute bin boundaries in logspace.
    kmids = np.zeros(np.size(kf)+1)
    kmids[1:-1] = np.exp((np.log(kf[1:])+np.log(kf[:-1]))/2.)
    #arbitrary final point
    kmids[-1] = 2*math.pi/2271 + kmids[-2]
    # This is the average of cos(2271k) across the k interval in the bin
    siform = np.zeros_like(kf)
    siform = (np.sin(2271*kmids[1:])-np.sin(2271*kmids[:-1]))/(kmids[1:]-kmids[:-1])/2271.
    #Correction for the zeroth bin, because the integral is oscillatory there.
    siform[0] = np.cos(2271*kf[0])
    return siform

def SiIIIcorr(fSiIII, tau_eff, kf):
    """The correction for SiIII contamination, as per McDonald."""
    assert tau_eff > 0
    aa = fSiIII/(1-np.exp(-tau_eff))
    return 1 + aa**2 + 2 * aa * _siIIIcorr(kf)

class LikelihoodClass(object):
    """Class to contain likelihood computations."""
    def __init__(self, basedir, datadir, mean_flux=True, nsamples=5000):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        #Parameter names
        sdss = gpemulator.SDSSData()
        myspec = flux_power.MySpectra(max_z=4.2)
        self.data_fluxpower = myspec.get_flux_power(datadir,sdss.get_kf(),tau0_factors=[0.8,])[0]
        #Use the SDSS covariance matrix
        self.data_covar = sdss.get_covar()
        self.emulator = coarse_grid.KnotEmulator(basedir)
        self.emulator.load()
        self.param_limits = self.emulator.get_param_limits(include_dense=mean_flux)
        self.gpemu = self.emulator.get_emulator(max_z=4.2, mean_flux=mean_flux)
        #Initialise sampler and make a few samples.
        self.sampler = self.init_emcee(nsamples=nsamples)

    def lnlike_linear(self, params):
        """A simple emcee likelihood function for the Lyman-alpha forest."""
        #Set parameter limits as the hull of the original emulator.
        if np.any(params < self.param_limits[:,0]) or np.any(params > self.param_limits[:,1]):
            return -np.inf
        predicted, std = self.gpemu.predict(params.reshape(1,-1))
        diff = predicted[0]-self.data_fluxpower
        #Ideally I would find a way to avoid this inversion
        icov = np.linalg.inv(self.data_covar + np.diag(std**2))
        return -np.dot(diff,np.dot(icov,diff))/2.0

    def init_emcee(self,nwalkers=100, burnin=500, nsamples = 5000):
        """Initialise and run emcee."""
        #Number of knots plus one cosmology plus one for mean flux.
        ndim = np.shape(self.param_limits)[0]
        #Limits: we need to hard-prior to the volume of our emulator.
        pr = (self.param_limits[:,1]-self.param_limits[:,0])
        #Priors are assumed to be in the middle.
        cent = (self.param_limits[:,1]+self.param_limits[:,0])/2.
        p0 = [cent+2*pr/16.*np.random.rand(ndim)-pr/16. for _ in range(nwalkers)]
        #assert np.all([np.isfinite(self.lnlike_linear(pp)) for pp in p0])
        emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnlike_linear)
        pos, _, _ = emcee_sampler.run_mcmc(p0, burnin)
        #Check things are reasonable
        assert np.all(emcee_sampler.acceptance_fraction > 0.01)
        emcee_sampler.reset()
        emcee_sampler.run_mcmc(pos, nsamples)
        return emcee_sampler

    def new_parameter_limits(self, all_samples, coverage=99):
        """Find a square region which includes coverage of the parameters in each direction, for refinement."""
        assert 50 < coverage < 100
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        new_par = np.percentile(all_samples,[100-coverage,coverage],axis=0)
        nsparse = np.shape(self.emulator.get_param_limits(include_dense=False))[0]
        return new_par.T[:nsparse,:]

    def refinement(self,nsamples,coverage=99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(self.sampler.flatchain,coverage=coverage)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

if __name__ == "__main__":
    like = LikelihoodClass(os.path.expanduser("~/data/Lya_Boss/cosmo-only-emulator"), os.path.expanduser("~/data/Lya_Boss/cosmo-only-test/AA0.94BB1.2CC0.71DD1.2hub0.71/output/"),mean_flux=True)
