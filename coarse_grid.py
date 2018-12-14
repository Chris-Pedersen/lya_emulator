"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import shutil
import string
import math
import json
import numpy as np
from SimulationRunner import simulationics
from SimulationRunner import lyasimulation
from SimulationRunner import clusters
import latin_hypercube
import flux_power
import matter_power
import lyman_data
import gpemulator
from mean_flux import ConstMeanFlux

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{s}', 'As': r'A_\mathrm{s}', 'heat_slope': r'H_\mathrm{S}', 'heat_amp': r'H_\mathrm{A}', 'hub':'h', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0'}
    try:
        return print_names[key]
    except KeyError:
        return key

class Emulator(object):
    """Small wrapper class to store parameter names and limits, generate simulations and get an emulator.
        Arguments:
            kf_bin_nums - list of element numbers of wavenumber array [kf] to be emulated. Default is all elements.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, kf=None, mf=None, kf_bin_nums=None):
        if param_names is None:
            self.param_names = {'ns':0, 'As':1, 'heat_slope':2, 'heat_amp':3, 'hub':4}
        else:
            self.param_names = param_names
        if param_limits is None:
            self.param_limits = np.array([[0.6, 1.5], [1.2e-9, 3.0e-9], [-0.5, 0.5],[0.5,1.5],[0.65,0.75]])
        else:
            self.param_limits = param_limits
        if kf is None:
            self.kf = lyman_data.BOSSData().get_kf(kf_bin_nums=kf_bin_nums)
        else:
            self.kf = kf
        if mf is None:
            self.mf = ConstMeanFlux(None)
        else:
            self.mf = mf
        #We fix omega_m h^2 = 0.1199 (Planck best-fit) and vary omega_m and h^2 to match it.
        #h^2 itself has no effect on the forest.
        self.omegamh2 = 0.1199
        #Corresponds to omega_m = (0.23, 0.31) which should be enough.
        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
        if not os.path.exists(basedir):
            os.mkdir(basedir)

    def build_dirname(self,params, include_dense=False):
        """Make a directory name for a given set of parameter values"""
        ndense = include_dense * len(self.mf.dense_param_names)
        parts = ['',]*(len(self.param_names) + ndense)
        #Transform the dictionary into a list of string parts,
        #sorted in the same way as the parameter array.
        if self.param_names == {'HeliumHeatAmp': 0}:
            string_formatting_type = '%.3g'
        else:
            string_formatting_type = '%.2g'
        for nn,val in self.mf.dense_param_names.items():
            parts[val] = nn+string_formatting_type % params[val]
        for nn,val in self.param_names.items():
            parts[ndense+val] = nn+string_formatting_type % params[ndense+val]
        name = ''.join(str(elem) for elem in parts)
        return name

    def print_pnames(self):
        """Get parameter names for printing"""
        n_latex = []
        sort_names = sorted(list(self.mf.dense_param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        sort_names = sorted(list(self.param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        return n_latex

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def dump(self, dumpfile="emulator_params.json"):
        """Dump parameters to a textfile."""
        #Backup existing parameter file
        fdump = os.path.join(self.basedir, dumpfile)
        if os.path.exists(fdump):
            backup = fdump + ".backup"
            r=1
            while os.path.exists(backup):
                backup = fdump + "_r"+str(r)+".backup"
                r+=1
            shutil.move(fdump, backup)
        #Arrays can't be serialised so convert them back and forth to lists
        self.really_arrays = []
        mf = self.mf
        self.mf = []
        for nn, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self.really_arrays.append(nn)
        with open(fdump, 'w') as jsout:
            json.dump(self.__dict__, jsout)
        self._fromarray()
        self.mf = mf

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        kf = self.kf
        mf = self.mf
        real_basedir = self.basedir
        with open(os.path.join(real_basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        self.kf = kf
        self.mf = mf
        self.basedir = real_basedir

    def get_outdir(self, pp):
        """Get the simulation output directory path for a parameter set."""
        return os.path.join(os.path.join(self.basedir, self.build_dirname(pp)),"output")

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def build_params(self, nsamples,limits = None, use_existing=False):
        """Build a list of directories and parameters from a hypercube sample"""
        if limits is None:
            limits = self.param_limits
        #Consider only prior points inside the limits
        prior_points = None
        if use_existing:
            ii = np.where(np.all(self.sample_params > limits[:,0],axis=1)*np.all(self.sample_params < limits[:,1],axis=1))
            prior_points = self.sample_params[ii]
        return latin_hypercube.get_hypercube_samples(limits, nsamples,prior_points=prior_points)

    def gen_simulations(self, nsamples, npart=256.,box=40,samples=None):
        """Initialise the emulator by generating simulations for various parameters."""
        if len(self.sample_params) == 0:
            self.sample_params = self.build_params(nsamples)
        if samples is None:
            samples = self.sample_params
        #Generate ICs for each set of parameter inputs
        for ev in samples:
            self._do_ic_generation(ev, npart, box)
        if samples is not None:
            self.sample_params = np.vstack([self.sample_params, samples])
        self.dump()
        return

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        #Use Planck 2015 cosmology
        ca={'rescale_gamma': True, 'rescale_slope': ev[pn['heat_slope']], 'rescale_amp' :ev[pn['heat_amp']]}
        hub = ev[pn['hub']]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to camb pivot scale
        ns = ev[pn['ns']]
        wmap = (2e-3/(2*math.pi/8.))**(ns-1.) * ev[pn['As']]
        ss = simulationics.SimulationICs(outdir=outdir, box=box,npart=npart, ns=ns, scalar_amp=wmap, hubble=hub, omega0=self.omegamh2/hub**2, omegab=0.0483, cluster_class=clusters.HypatiaClass) #MP #code_args = ca #code_class=lyasimulation.LymanAlphaSim
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

    def get_param_limits(self, include_dense=True):
        """Get the reprocessed limits on the parameters for the likelihood."""
        if not include_dense:
            return self.param_limits
        dlim = self.mf.get_limits()
        if dlim is not None:
            #Dense parameters go first as they are 'slow'
            plimits = np.vstack([dlim, self.param_limits])
            assert np.shape(plimits)[1] == 2
            return plimits
        return self.param_limits

    def get_nsample_params(self):
        """Get the number of sparse parameters, those sampled by simulations."""
        return np.shape(self.param_limits)[0]

    def _get_fv(self, pp,myspec):
        """Helper function to get a single flux vector."""
        di = self.get_outdir(pp)
        powerspectra = myspec.get_snapshot_list(base=di)
        return powerspectra

    def get_emulator(self, max_z=4.2):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        gp = self._get_custom_emulator(emuobj=gpemulator.MultiBinGP, max_z=max_z)
        return gp

    def get_flux_vectors(self, max_z=4.2):
        """Get the desired flux vectors and their parameters"""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        assert nparams == len(self.param_names)
        myspec = flux_power.MySpectra(max_z=max_z)
        powers = [self._get_fv(pp, myspec) for pp in pvals]
        mean_fluxes = np.exp(-self.mf.get_t0(myspec.zout))
        #Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = self.mf.get_params()
        flux_vectors = np.array([ps.get_power(kf = self.kf, mean_fluxes = mef) for mef in mean_fluxes for ps in powers])
        if dpvals is not None:
            aparams = np.array([np.concatenate([dp,pv]) for dp in dpvals for pv in pvals])
        else:
            aparams = pvals
        return aparams, flux_vectors

    def _get_custom_emulator(self, *, emuobj, max_z=4.2):
        """Helper to allow supporting different emulators."""
        aparams, flux_vectors = self.get_flux_vectors(max_z=max_z)
        plimits = self.get_param_limits(include_dense=True)
        gp = emuobj(params=aparams, kf=self.kf, powers = flux_vectors, param_limits = plimits)
        return gp

class KnotEmulator(Emulator):
    """Specialise parameter class for an emulator using knots.
    Thermal parameters turned off."""
    def __init__(self, basedir, nknots=4, kf=None, mf=None):
        param_names = {'heat_slope':nknots, 'heat_amp':nknots+1, 'hub':nknots+2}
        #Assign names like AA, BB, etc.
        for i in range(nknots):
            param_names[string.ascii_uppercase[i]*2] = i
        self.nknots = nknots
        param_limits = np.append(np.repeat(np.array([[0.6,1.5]]),nknots,axis=0),[[-0.5, 0.5],[0.5,1.5],[0.65,0.75]],axis=0)
        super().__init__(basedir=basedir, param_names = param_names, param_limits = param_limits, kf=kf, mf=mf)
        #Linearly spaced knots in k space:
        #these do not quite hit the edges of the forest region, because we want some coverage over them.
        self.knot_pos = np.linspace(0.15, 1.5,nknots)
        #Used for early iterations.
        #self.knot_pos = [0.15,0.475,0.75,1.19]

    def _do_ic_generation_knots(self,ev,npart,box): #Potentially risky to rename this
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        #Use Planck 2015 cosmology
        ca={'rescale_gamma': True, 'rescale_slope': ev[pn['heat_slope']], 'rescale_amp' :ev[pn['heat_amp']]}
        hub = ev[pn['hub']]
        ss = lyasimulation.LymanAlphaKnotICs(outdir=outdir, box=box,npart=npart, knot_pos = self.knot_pos, knot_val=ev[0:self.nknots],hubble=hub, code_class=lyasimulation.LymanAlphaMPSim, code_args = ca, omega0=self.omegamh2/hub**2, omegab=0.0483)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

class MatterPowerEmulator(Emulator):
    """Build an emulator based on the matter power spectrum instead of the flux power spectrum, for testing."""
    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile. Reset the k values to something sensible for matter power."""
        super().load(dumpfile=dumpfile)
        self.kf = np.logspace(np.log10(3*math.pi/60.),np.log10(2*math.pi/60.*256),20)

    def _get_fv(self, pp,myspec):
        """Helper function to get a single matter power vector."""
        di = self.get_outdir(pp)
        (_,_) = myspec
        fv = matter_power.get_matter_power(di,kk=self.kf, redshift = 3.)
        return fv

def make_emulator_latin_hypercube(emulator_directory, n_simulations, parameter_limits, omegamh2=0.1327, hypatia_queue='cores24', prior_points=None, refinement=False, json_file_name='/emulator_params.json'):
    """Small wrapper to make Latin hypercube emulator"""
    simulation_parameters = latin_hypercube.get_hypercube_samples(parameter_limits, n_simulations, prior_points=prior_points)
    print('New simulation parameters =', simulation_parameters)
    #np.save('new_params.npy', simulation_parameters)
    if refinement:
        simulation_parameters = simulation_parameters[prior_points.shape[0]:]
    generate_emulator_submissions(emulator_directory, simulation_parameters, parameter_limits, omegamh2=omegamh2, hypatia_queue=hypatia_queue, refinement=refinement, json_file_name=json_file_name)

def generate_emulator_submissions(emulator_directory, simulation_parameters, parameter_limits, omegamh2=0.1327, hypatia_queue='cores24', refinement=False, json_file_name='/emulator_params.json'):
    """Small function to generate directory structure and submission files for an emulator"""
    #gadget_parameters = latin_hypercube.convert_to_simulation_parameters(simulation_parameters, omegamh2=omegamh2)
    if hypatia_queue == 'cores24':
        pbs_file_name = '/run.pbs'
    elif hypatia_queue == 'cores12':
        pbs_file_name = '/run_cores12.pbs'
    elif hypatia_queue == 'cores40':
        pbs_file_name = '/run_cores40.pbs'
    elif hypatia_queue == 'smp':
        pbs_file_name = '/run_smp.pbs'

    default_files_directory = '/share/data2/keir/Simulations'
    genic_file_name = '/paramfile.genic'
    gadget_file_name = '/paramfile.gadget'
    #json_file_name = '/emulator_params.json'
    class_file = default_files_directory + '/make_class_power.py'

    for i in range(simulation_parameters.shape[0]): #Loop over simulations
        gadget_parameters = latin_hypercube.convert_to_simulation_parameters(simulation_parameters[i], omegamh2=omegamh2)

        simulation_directory = emulator_directory + '/ns%.2gAs%.2gheat_slope%.2gheat_amp%.2ghub%.2g' % tuple(simulation_parameters[i])
        os.makedirs(simulation_directory)
        os.makedirs(simulation_directory + '/output')

        shutil.copyfile(default_files_directory + pbs_file_name, simulation_directory + pbs_file_name)

        new_genic_file1 = simulation_directory + '/paramfile_genic.genic'
        shutil.copyfile(default_files_directory + genic_file_name, new_genic_file1)
        with open(new_genic_file1, 'a') as new_genic_file_object1:
            new_genic_file_object1.write('Omega0 = %.6g\n' % gadget_parameters['Omega0'])
            new_genic_file_object1.write('OmegaLambda = %.6g\n' % gadget_parameters['OmegaLambda'])
            new_genic_file_object1.write('OmegaBaryon = %.6g\n' % gadget_parameters['OmegaBaryon'])
            new_genic_file_object1.write('HubbleParam = %.6g\n' % gadget_parameters['HubbleParam'])
            new_genic_file_object1.write('PrimordialIndex = %.6g\n' % gadget_parameters['PrimordialIndex'])
            new_genic_file_object1.write('PrimordialAmp = %.6g\n' % gadget_parameters['PrimordialAmp'])

        new_genic_file2 = simulation_directory + genic_file_name #For make_class_power.py
        shutil.copyfile(new_genic_file1, new_genic_file2)
        with open(new_genic_file2, 'a') as new_genic_file_object2:
            new_genic_file_object2.write('InputFutureRedshift = 98.99\n')
            new_genic_file_object2.write('FileWithFutureTransferFunction = transfer_future.dat\n')

        new_gadget_file = simulation_directory + gadget_file_name
        shutil.copyfile(default_files_directory + gadget_file_name, new_gadget_file)
        with open(new_gadget_file, 'a') as new_gadget_file_object:
            new_gadget_file_object.write('Omega0 = %.6g\n' % gadget_parameters['Omega0'])
            new_gadget_file_object.write('OmegaLambda = %.6g\n' % gadget_parameters['OmegaLambda'])
            new_gadget_file_object.write('OmegaBaryon = %.6g\n' % gadget_parameters['OmegaBaryon'])
            new_gadget_file_object.write('HubbleParam = %.6g\n' % gadget_parameters['HubbleParam'])
            new_gadget_file_object.write('HeliumHeatAmp = %.6g\n' % simulation_parameters[i][3])
            new_gadget_file_object.write('HeliumHeatExp = %.6g\n' % simulation_parameters[i][2])

        os.chdir(simulation_directory)
        os.system('python %s %s' % (class_file, genic_file_name[1:]))
        '''with open(class_file, 'r') as class_file_object:
            code = compile(class_file_object.read(), class_file, 'exec')
            exec(code, )'''
        #os.system('qsub %s' % pbs_file_name[1:])

    new_json_file = emulator_directory + json_file_name
    if not refinement:
        shutil.copyfile(default_files_directory + json_file_name, new_json_file)
    with open(new_json_file, 'r') as new_json_file_object:
        json_dictionary = json.load(new_json_file_object)
    json_dictionary['param_limits'] = parameter_limits.tolist()
    json_dictionary['omegamh2'] = omegamh2
    if not refinement:
        json_dictionary['sample_params'] = simulation_parameters.tolist()
    else:
        json_dictionary['sample_params'] += simulation_parameters.tolist()
    json_dictionary['basedir'] = emulator_directory
    with open(new_json_file, 'w') as new_json_file_object:
        json.dump(json_dictionary, new_json_file_object)

#def submit_emulator_submissions()
