"""Module for computing the likelihood function for the forest emulator."""
import os
import os.path
import math
import numpy as np
import numpy.linalg as npl
import numpy.testing as npt
import emcee
import coarse_grid
import flux_power
import lyman_data
import mean_flux as mflux
import scipy.integrate as spg
import scipy.interpolate as spi
#from datetime import datetime

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

def gelman_rubin(chain):
    """Compute the Gelman-Rubin statistic for a chain"""
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    tb = np.mean(chain, axis=1)
    tbb = np.mean(tb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((tbb - tb)**2, axis=0)
    var_t = (n - 1) / n * W + 1 / n * B
    R = np.sqrt(var_t / W)
    return R

def invert_block_diagonal_covariance(full_covariance_matrix, n_blocks):
    """Efficiently invert block diagonal covariance matrix"""
    inverse_covariance_matrix = np.zeros_like(full_covariance_matrix)
    nz = n_blocks
    nk = int(full_covariance_matrix.shape[0] / nz)
    for z in range(nz): #Loop over blocks by redshift
        start_index = nk * z
        end_index = nk * (z + 1)
        inverse_covariance_block = npl.inv(full_covariance_matrix[start_index: end_index, start_index: end_index])
        inverse_covariance_matrix[start_index: end_index, start_index: end_index] = inverse_covariance_block
    return inverse_covariance_matrix


class LikelihoodClass(object):
    """Class to contain likelihood computations."""
    def __init__(self, basedir, datadir, mean_flux='s', max_z = 4.2, t0_training_value = 0.95, rescale_data_error=False, fix_error_ratio=False, error_ratio=100.):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        self.rescale_data_error = rescale_data_error
        self.fix_error_ratio = fix_error_ratio
        self.error_ratio = error_ratio

        self._inverse_BOSS_covariance_full = None

        #Use the BOSS covariance matrix
        self.sdss = lyman_data.BOSSData()
        #'Data' now is a simulation
        self.max_z = max_z
        myspec = flux_power.MySpectra(max_z=self.max_z)
        self.zout = myspec.zout
        #print(datadir)
        pps = myspec.get_snapshot_list(datadir)
        self.kf = self.sdss.get_kf()

        #Load BOSS data vector
        self.BOSS_flux_power = self.sdss.pf.reshape(-1, self.kf.shape[0])[:self.zout.shape[0]][::-1] #km / s; n_z * n_k

        self.data_fluxpower = pps.get_power(kf=self.kf, mean_fluxes=np.exp(-mflux.obs_mean_tau(self.zout, amp=0) * t0_training_value))
        assert np.size(self.data_fluxpower) % np.size(self.kf) == 0
        self.mf_slope = False
        #Param limits on t0
        t0_factor = np.array([0.75,1.25])
        #Get the emulator
        if mean_flux == 'c':
            mf = mflux.ConstMeanFlux(value = t0_training_value)
        #As each redshift bin is independent, for redshift-dependent mean flux models
        #we just need to convert the input parameters to a list of mean flux scalings
        #in each redshift bin.
        #This is an example which parametrises the mean flux as an amplitude and slope.
        elif mean_flux == 's':
            #Add a slope to the parameter limits
            t0_slope =  np.array([-0.25, 0.25])
            self.mf_slope = True
            slopehigh = np.max(mflux.mean_flux_slope_to_factor(np.linspace(2.2, max_z, 11),0.25))
            slopelow = np.min(mflux.mean_flux_slope_to_factor(np.linspace(2.2, max_z, 11),-0.25))
            dense_limits = np.array([np.array(t0_factor) * np.array([slopelow, slopehigh])])
            mf = mflux.MeanFluxFactor(dense_limits = dense_limits)
        self.emulator = coarse_grid.KnotEmulator(basedir, kf=self.kf, mf=mf)
        self.emulator.load()
        self.param_limits = self.emulator.get_param_limits(include_dense=True)
        if mean_flux == 's':
            #Add a slope to the parameter limits
            self.param_limits = np.vstack([t0_slope, self.param_limits])
            #Shrink param limits t0 so that even with
            #a slope they are within the emulator range
            self.param_limits[1,:] = t0_factor
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2
        #print('Beginning to generate emulator at', str(datetime.now()))
        self.gpemu = self.emulator.get_emulator(max_z=max_z)
        #print('Finished generating emulator at', str(datetime.now()))

    def likelihood(self, params, include_emu=True):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix."""
        nparams = params
        if self.mf_slope:

            # tau_0_i[z] @dtau_0 / tau_0_i[z] @[dtau_0 = 0]
            # Divided by lowest redshift case
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])

            nparams = params[1:] #Keep only t0 sampling parameter (of mean flux parameters)
        else: #Otherwise bug if choose mean_flux = 'c'
            tau0_fac = None
        if np.any(params >= self.param_limits[:,1]) or np.any(params <= self.param_limits[:,0]):
            return -np.inf
        #Set parameter limits as the hull of the original emulator.

        # .predict should take [{list of parameters: t0; cosmo.; thermal},]
        # Here: emulating @ cosmo.; thermal; sampled t0 * [tau0_fac from above]
        predicted, std = self.gpemu.predict(np.array(nparams).reshape(1,-1), tau0_factors = tau0_fac)

        #Save emulated flux power specra for analysis
        self.emulated_flux_power = predicted
        self.emulated_flux_power_std = std

        diff = predicted[0]-self.data_fluxpower
        nkf = len(self.kf)
        nz = int(len(diff)/nkf)
        #Likelihood using full covariance matrix
        chi2 = 0
        #Redshifts
        sdssz = self.sdss.get_redshifts()

        #Fix maximum redshift bug
        sdssz = sdssz[sdssz <= self.max_z]

        #Important assertion
        assert nz == sdssz.size
        npt.assert_allclose(sdssz, self.zout, atol=1.e-16)
        #print('SDSS redshifts are', sdssz)

        self.exact_flux_power_std = [None] * nz
        for bb in range(nz):
            diff_bin = diff[nkf*bb:nkf*(bb+1)]
            std_bin = std[0,nkf*bb:nkf*(bb+1)]
            #covar_bin = self.sdss.get_covar(sdssz[bb])

            #Rescale mock measurement covariance matrix to match BOSS percentage accuracy
            '''if self.rescale_data_error:
                rescaling_factor = self.data_fluxpower[nkf*bb:nkf*(bb+1)] / self.BOSS_flux_power[bb] #Rescale 1 sigma
                covar_bin *= np.outer(rescaling_factor, rescaling_factor) #(km / s)**2'''
            covar_bin = self.get_BOSS_covariance_single_z(self.zout[bb])

            if self.fix_error_ratio:
                fix_rescaling_factor = self.error_ratio * np.mean(std_bin) / np.mean(np.sqrt(np.diag(covar_bin)))
                covar_bin *= np.outer(fix_rescaling_factor, fix_rescaling_factor)
            self.exact_flux_power_std[bb] = np.sqrt(np.diag(covar_bin))

            assert np.shape(np.diag(std_bin**2)) == np.shape(covar_bin)
            if include_emu:
                #Assume each k bin is independent
                covar_bin += np.diag(std_bin**2)
                #Assume completely correlated emulator errors within this bin
#                 covar_bin += np.matmul(np.diag(std_bin**2),np.ones_like(covar_bin))
            icov_bin = np.linalg.inv(covar_bin)
            (_, cdet) = np.linalg.slogdet(covar_bin)
            dcd = - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
            chi2 += dcd -0.5* cdet
            assert 0 > chi2 > -2**31
            assert not np.isnan(chi2)
        return chi2

    def likelihood_marginalised_mean_flux(self, params, include_emu=True, integration_options=None, verbose=True): #marginalised_axes=(0, 1)
        """Evaluate (Gaussian) likelihood marginalised over mean flux parameter axes: (dtau0, tau0)"""
        #assert len(marginalised_axes) == 2
        assert self.mf_slope
        likelihood_function = lambda dtau0, tau0: self.likelihood(np.concatenate(([dtau0,], [tau0,], params)), include_emu=include_emu)
        integration_output = spg.nquad(likelihood_function, self.param_limits[:2], opts=integration_options, full_output=verbose)
        print(integration_output)
        return integration_output[0]

    def get_BOSS_covariance_single_z(self, redshift):
        """Get the BOSS covariance matrix at a given redshift"""
        redshift_index = np.nonzero(self.zout == redshift)[0][0]
        covariance_unscaled = self.sdss.get_covar(redshift)
        if not self.rescale_data_error:
            return covariance_unscaled
        else:
            nkf = len(self.kf)
            rescaling_factor = self.data_fluxpower[nkf*redshift_index:nkf*(redshift_index+1)] / self.BOSS_flux_power[redshift_index]
            return covariance_unscaled * np.outer(rescaling_factor, rescaling_factor)

    def get_BOSS_covariance_full(self):
        """Get the full BOSS covariance matrix (for all redshifts)"""
        nkf = len(self.kf)
        nz = len(self.zout)
        n_data_points = nkf * nz
        covariance_full = np.zeros((n_data_points, n_data_points))
        for i in range(self.zout.shape[0]): #Loop over redshifts (in descending order)
            start_index = i * nkf
            end_index = (i+1) * nkf
            covariance_full[start_index:end_index, start_index:end_index] = self.get_BOSS_covariance_single_z(self.zout[i])
        return covariance_full

    def get_inverse_BOSS_covariance_full(self):
        """Get the inverse of the full BOSS covariance matrix (for all redshifts)"""
        if self._inverse_BOSS_covariance_full is None:
            self._inverse_BOSS_covariance_full = invert_block_diagonal_covariance(self.get_BOSS_covariance_full(), self.zout.shape[0])
        return self._inverse_BOSS_covariance_full

    def do_sampling(self, savefile, nwalkers=100, burnin=5000, nsamples=5000, while_loop=True, include_emulator_error=True):
        """Initialise and run emcee."""
        pnames = self.emulator.print_pnames()
        if self.mf_slope:
            pnames = [('dtau0',r'd\tau_0'),]+pnames
        with open(savefile+"_names.txt",'w') as ff:
            for pp in pnames:
                ff.write("%s %s\n" % pp)
        #Limits: we need to hard-prior to the volume of our emulator.
        pr = (self.param_limits[:,1]-self.param_limits[:,0])
        #Priors are assumed to be in the middle.
        cent = (self.param_limits[:,1]+self.param_limits[:,0])/2.
        p0 = [cent+2*pr/16.*np.random.rand(self.ndim)-pr/16. for _ in range(nwalkers)]
        assert np.all([np.isfinite(self.likelihood(pp, include_emu=include_emulator_error)) for pp in p0])
        emcee_sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.likelihood, args=(include_emulator_error,))
        pos, _, _ = emcee_sampler.run_mcmc(p0, burnin)
         #Check things are reasonable
        assert np.all(emcee_sampler.acceptance_fraction > 0.01)
        emcee_sampler.reset()
        self.cur_results = emcee_sampler
        gr = 10.
        while np.any(gr > 1.05):
            emcee_sampler.run_mcmc(pos, nsamples)
            gr = gelman_rubin(emcee_sampler.chain)
            print("Total samples:",nsamples," Gelman-Rubin: ",gr)
            np.savetxt(savefile, emcee_sampler.flatchain)
            if while_loop is False:
                break
        return emcee_sampler

    def new_parameter_limits(self, confidence=0.99, include_dense=False):
        """Find a square region which includes coverage of the parameters in each direction, for refinement.
        Confidence must be 0.68, 0.95 or 0.99."""
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        #Get marginalised statistics.
        limits = np.percentile(self.flatchain, [100-100*confidence, 100*confidence], axis=0).T
        #Discard dense params
        ndense = len(self.emulator.mf.dense_param_names)
        if self.mf_slope:
            ndense+=1
        if include_dense:
            ndense = 0
        lower = limits[ndense:,0]
        upper = limits[ndense:, 1]
        assert np.all(lower < upper)
        new_par = limits[ndense:,:]
        return new_par

    def get_covar_det(self, params, include_emu):
        """Get the determinant of the covariance matrix.for certain parameters"""
        nparams = params
        if self.mf_slope:
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])
            nparams = params[1:]
        else: #Otherwise bug if choose mean_flux = 'c'
            tau0_fac = None
        if np.any(params >= self.param_limits[:,1]) or np.any(params <= self.param_limits[:,0]):
            return -np.inf
        sdssz = self.sdss.get_redshifts()
        #Fix maximum redshift bug
        sdssz = sdssz[sdssz <= self.max_z]
        nz = sdssz.size
        nkf = len(self.kf)
        if include_emu:
            _, std = self.gpemu.predict(np.array(nparams).reshape(1,-1), tau0_factors = tau0_fac)
        detc = 1
        for bb in range(nz):
            covar_bin = self.sdss.get_covar(sdssz[bb])
            if include_emu:
                std_bin = std[0,nkf*bb:nkf*(bb+1)]
                #Assume completely correlated emulator errors within this bin
                covar_emu = np.outer(std_bin, std_bin)
                covar_bin += covar_emu
            _, det_bin = np.linalg.slogdet(covar_bin)
            #We have a block diagonal covariance
            detc *= det_bin
        return detc

    def refine_metric(self, params, use_error_ratio=False):
        """This evaluates the 'refinement metric':
           the extent to which the emulator error dominates the covariance.
           The idea is that when it is > 1, refinement is necessary"""
        if not use_error_ratio:
            detnoemu = self.get_covar_det(params, False)
            detemu = self.get_covar_det(params, True)
            return detemu/detnoemu
        else:
            likelihood_evaluation = self.likelihood(params)
            assert self.emulated_flux_power_std[0].size == np.array(self.exact_flux_power_std).size
            return np.mean(self.emulated_flux_power_std[0]) / np.mean(np.array(self.exact_flux_power_std))

    def _get_emulator_error_averaged_mean_flux(self, params):
        """Get the emulator error having averaged over the mean flux parameter axes: (dtau0, tau0)"""
        n_samples = 10
        emulator_error_total = 0.
        for dtau0 in np.linspace(self.param_limits[0, 0], self.param_limits[0, 1], num=n_samples):
            for tau0 in np.linspace(self.param_limits[1, 0], self.param_limits[1, 1], num=n_samples):
                _ = self.likelihood(np.concatenate(([dtau0,], [tau0,], params)))
                emulator_error_total += self.emulated_flux_power_std[0]
        return emulator_error_total / (n_samples ** 2)

    def _get_GP_UCB_exploitation_term(self, objective_function, exploitation_weight=1.):
        """Evaluate the exploitation term of the GP-UCB acquisition function"""
        return objective_function * exploitation_weight

    def _get_GP_UCB_exploration_term(self, data_vector_emulator_error, n_emulated_params, iteration_number=1, delta=0.5, nu=1.):
        """Evaluate the exploration term of the GP-UCB acquisition function"""
        exploration_weight = math.sqrt(nu * 2. * math.log((iteration_number**((n_emulated_params / 2.) + 2.)) * (math.pi**2) / 3. / delta))
        inverse_measurement_covariance = self.get_inverse_BOSS_covariance_full()
        posterior_estimated_error = np.dot(data_vector_emulator_error, np.dot(inverse_measurement_covariance, data_vector_emulator_error))
        return exploration_weight * posterior_estimated_error

    def acquisition_function_GP_UCB(self, params, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1.):
        """Evaluate the GP-UCB at given parameter vector. This is an acquisition function for determining where to run
        new training simulations"""
        assert iteration_number >= 1.
        assert 0. < delta < 1.
        if self.mf_slope:
            n_emulated_params = params.shape[0] - 1
        else:
            n_emulated_params = params.shape[0]
        #exploitation_term = self.likelihood(params) * exploitation_weight #Log-posterior [weighted]

        exploitation = self._get_GP_UCB_exploitation_term(self.likelihood(params), exploitation_weight)
        exploration = self._get_GP_UCB_exploration_term(self.emulated_flux_power_std[0], n_emulated_params, iteration_number=iteration_number, delta=delta, nu=nu)
        return exploitation + exploration

    def acquisition_function_GP_UCB_marginalised_mean_flux(self, params, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., integration_options=None):
        """Evaluate the GP-UCB acquisition function, having marginalised over mean flux parameter axes: (dtau0, tau0)"""
        #if exploitation_weight is None:
        print('No exploitation term')
        exploitation = 0.
        #else:
        #    exploitation = self._get_GP_UCB_exploitation_term(self.likelihood_marginalised_mean_flux(params, integration_options=integration_options), exploitation_weight=exploitation_weight)
        exploration = self._get_GP_UCB_exploration_term(self._get_emulator_error_averaged_mean_flux(params), params.size, iteration_number=iteration_number, delta=delta, nu=nu)
        return exploitation + exploration

    def check_for_refinement(self, conf = 0.95, thresh = 1.05):
        """Crude check for refinement: check whether the likelihood is dominated by
           emulator error at the 1 sigma contours."""
        limits = self.new_parameter_limits(confidence=conf, include_dense = True)
        while True:
            #Do the check
            uref = self.refine_metric(limits[:,0])
            lref = self.refine_metric(limits[:,1])
            #This should be close to 1.
            print("up =",uref," low=",lref)
            if (uref < thresh) and (lref < thresh):
                break
            #Iterate by moving each limit 40% outwards.
            midpt = np.mean(limits, axis=1)
            limits[:,0] = 1.4*(limits[:,0] - midpt) + midpt
            limits[:,0] = np.max([limits[:,0], self.param_limits[:,0]],axis=0)
            limits[:,1] = 1.4*(limits[:,1] - midpt) + midpt
            limits[:,1] = np.min([limits[:,1], self.param_limits[:,1]],axis=0)
            if np.all(limits == self.param_limits):
                break
        return limits

    def refinement(self,nsamples,confidence=0.99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(confidence=confidence)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        assert np.shape(new_samples)[0] == nsamples
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

    def _interpolate_err_grid(self, i, j, rsamples, randscores):
        """Interpolate samples onto a 2D grid"""
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        grid_x = grid_x * (self.param_limits[i,1] - self.param_limits[i,0]) + self.param_limits[i,0]
        grid_y = grid_y * (self.param_limits[j,1] - self.param_limits[j,0]) + self.param_limits[j,0]
        return spi.griddata(rsamples[:,(i,j)], randscores,(grid_x,grid_y),fill_value = 0)

    def make_err_grid(self, i, j, samples = 30000, use_error_ratio=False):
        """Make an error grid"""
        ndim = np.size(self.param_limits[:,0])
        rr = lambda x : np.random.rand(ndim)*(self.param_limits[:,1]-self.param_limits[:,0]) + self.param_limits[:,0]
        rsamples = np.array([rr(i) for i in range(samples)])
        randscores = [self.refine_metric(rr, use_error_ratio=use_error_ratio) for rr in rsamples]
        return self._interpolate_err_grid(i, j, rsamples, randscores)

    def _get_parameter_grid_single_slice(self, i, j, random_samples=True, samples=30000):
        """Get a grid of parameter values on a 2D slice through the hyper-volume"""
        if not random_samples:
            samples = 40000
        rsamples = np.ones((samples, np.size(self.param_limits[:,0]))) * 0.5
        if random_samples:
            rsamples[:, i] = np.random.rand(samples)
            rsamples[:, j] = np.random.rand(samples)
        else:
            slice_samples = np.mgrid[0:1:200j, 0:1:200j]
            rsamples[:, i] = slice_samples[0].flatten()
            rsamples[:, j] = slice_samples[1].flatten()
        return rsamples * (self.param_limits[np.newaxis, :, 1] - self.param_limits[np.newaxis, :, 0]) + self.param_limits[np.newaxis, :, 0]

    def make_err_grid_single_slice(self, i, j, samples=30000, use_error_ratio=False):
        """Make an error grid on just a single slice through the hyper-volume"""
        rsamples = self._get_parameter_grid_single_slice(i, j, samples=samples)
        randscores = [self.refine_metric(rr, use_error_ratio=use_error_ratio) for rr in rsamples]
        return self._interpolate_err_grid(i, j, rsamples, randscores)

    def make_grid_acquisition_function(self, i, j, random_samples=True, samples=30000, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., mean_flux_marginalisation=True, likelihood_integration_options=None):
        """Make a grid of acquisition function evaluations on a 2D slice through the hyper-volume"""
        parameter_samples = self._get_parameter_grid_single_slice(i, j, random_samples=random_samples, samples=samples)
        if mean_flux_marginalisation:
            acquisition_function = lambda x: self.acquisition_function_GP_UCB_marginalised_mean_flux(x[2:], iteration_number=iteration_number, delta=delta, nu=nu, exploitation_weight=exploitation_weight, integration_options=likelihood_integration_options)
        else:
            acquisition_function = lambda x: self.acquisition_function_GP_UCB(x, iteration_number=iteration_number, delta=delta, nu=nu, exploitation_weight=exploitation_weight)
        print('Evaluating acquisition function at parameter samples')
        acquisition_samples = np.array([acquisition_function(parameter_vector) for parameter_vector in parameter_samples])
        print('Forming grid of acquisition function evaluations')
        if random_samples:
            return self._interpolate_err_grid(i, j, parameter_samples, acquisition_samples)
        else:
            return acquisition_samples.reshape((int(math.sqrt(acquisition_samples.shape[0])), -1))

if __name__ == "__main__":
    like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots_refine"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots_test/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output"))
    #Works very well!
    #     like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots/AA0.96BB1.3CC1DD1.3heat_slope-5.6e-17heat_amp1.2hub0.66/output"))
    #output = like.do_sampling(os.path.expanduser("~/Simulations/Lya_Boss/hires_knots_test/AA0.97BB1.3_chain.txt"))
