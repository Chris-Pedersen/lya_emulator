"""Module to load the covariance matrix (from BOSS DR9 or SDSS DR5 data) from tables."""
import os.path
import numpy as np
import numpy.testing as npt

class SDSSData(object):
    """A class to store the flux power and corresponding covariance matrix from SDSS. A little tricky because of the redshift binning."""
    def __init__(self, datafile="data/lya.sdss.table.txt", covarfile="data/lya.sdss.covar.txt"):
        # Read SDSS best-fit data.
        # Contains the redshift wavenumber from SDSS
        # See 0405013 section 5.
        # First column is redshift
        # Second is k in (km/s)^-1
        # Third column is P_F(k)
        # Fourth column (ignored): square roots of the diagonal elements
        # of the covariance matrix. We use the full covariance matrix instead.
        # Fifth column (ignored): The amount of foreground noise power subtracted from each bin.
        # Sixth column (ignored): The amound of background power subtracted from each bin.
        # A metal contamination subtraction that McDonald does but we don't.
        data = np.loadtxt(datafile)
        self.redshifts = data[:,0]
        self.kf = data[:,1]
        self.pf = data[:,1]
        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())
        assert self.nz * self.nk == np.size(self.kf)
        #The covariance matrix, correlating each k and z bin with every other.
        #kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.loadtxt(covarfile)
        self.covar_diag = data[:, 3] ** 2

    def get_kf(self, kf_bin_nums=None):
        """Get the (unique) flux k values"""
        kf_array = np.sort(np.array(list(set(self.kf))))
        if kf_bin_nums is None:
            return kf_array
        else:
            return kf_array[kf_bin_nums]

    def get_redshifts(self):
        """Get the (unique) redshift bins, sorted in decreasing redshift"""
        return np.sort(np.array(list(set(self.redshifts))))[::-1]

    def get_pf(self, zbin=None):
        """Get the power spectrum"""
        if zbin is None:
            return self.pf
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01))
        return self.pf[ii]

    def get_icovar(self):
        """Get the inverse covariance matrix"""
        return np.linalg.inv(self.covar)

    def get_covar(self, zbin=None):
        """Get the correlation matrix"""
        _ = zbin
        return self.covar

    def get_covar_diag(self):
        """Get the (diagonal of the) covariance matrix"""
        return self.covar_diag


class BOSSData(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from BOSS."""
    def __init__(self, datafile=None, covardir=None):

        cdir = os.path.dirname(__file__)
        if datafile is None:
            datafile = os.path.join(cdir,"data/boss_dr9_data/table4a.dat")
        if covardir is None:
            covardir = os.path.join(cdir, "data/boss_dr9_data")
        # Read SDSS best-fit data.
        # Contains the redshift wavenumber from SDSS
        # See Readme file.
        # Fourth column: square roots of covariance diagonal
        data = np.loadtxt(datafile)
        self.redshifts = data[:,2]
        self.kf = data[:,3]
        self.pf = data[:,4]
        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())
        assert self.nz * self.nk == np.size(self.kf)
        self.covar_diag = data[:,5]**2 + data[:,8]**2
        #The covariance matrix, correlating each k and z bin with every other.
        #kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.zeros((len(self.redshifts),len(self.redshifts))) #Full covariance matrix (35*12 x 35*12) for k,z
        for bb in range(12):
            dfile = os.path.join(covardir,"cct4b"+str(bb+1)+".dat")
            dd = np.loadtxt(dfile) #k-bin covariance matrix (35 x 35) for single redshift
            self.covar[35*bb:35*(bb+1),35*bb:35*(bb+1)] = dd #Filling in block matrices along diagonal

    def get_covar(self, zbin=None):
        """Get the covariance matrix"""
        if zbin is None:
            std_diag = np.sqrt(self.covar_diag)
            return self.covar * np.outer((std_diag, std_diag))
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01)) #Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii)+1)
        #return self.covar[rr[0]:rr[1],rr[0]:rr[1]] * self.covar_diag[rr[0]:rr[1]]
        #Bug fix
        std_diag_single_z = np.sqrt(self.covar_diag[rr[0]:rr[1]])
        covar_matrix = self.covar[rr[0]:rr[1], rr[0]:rr[1]] * np.outer(std_diag_single_z, std_diag_single_z)
        npt.assert_allclose(np.diag(covar_matrix), self.covar_diag[rr[0]:rr[1]], atol=1.e-16)
        return covar_matrix


class BoeraData(SDSSData):
    """A class to store the flux power spectrum and covariance matrix from Boera+ 2018 (HIRES/UVES;
    arxiv:1809.06980)."""
    def __init__(self, datadir=None, covardir=None):
        file_directory = os.path.dirname(__file__)
        if datadir is None:
            datadir = os.path.join(file_directory, 'data/Boera_HIRES_UVES_flux_power')
        if covardir is None:
            covardir = os.path.join(datadir, 'apjaafee4')

        self.redshifts_unique = np.array([4.24, 4.58, 4.95])
        self.nz = self.redshifts_unique.size
        self.nk = 16
        self.redshifts = np.repeat(self.redshifts_unique, self.nk)
        self.kf = np.zeros_like(self.redshifts)
        self.pk = np.zeros_like(self.redshifts)
        self.covar_diag = np.zeros_like(self.redshifts)
        self.covar = np.zeros((self.nk * self.nz, self.nk * self.nz))
        self.covar_full = np.zeros((self.nz, self.nk * self.nz))
        assert self.nz * self.nk == self.kf.size

        for i in range(self.nz):
            flux_power_file = os.path.join(datadir, 'flux_power_z_%.1f.dat'%self.redshifts_unique[i])
            flux_power_data = np.genfromtxt(flux_power_file, skip_header=5, skip_footer=1)

            start_index = i * self.nk
            end_index = (i + 1) * self.nk
            self.kf[start_index: end_index] = 10 ** flux_power_data[:, 0]
            self.pk[start_index: end_index] = flux_power_data[:, 2]
            self.covar_diag[start_index: end_index] = flux_power_data[:, 3] ** 2

            covar_file = os.path.join(covardir, 'Cov_Matrixz=%.1f.dat'%self.redshifts_unique[i])
            self.covar_full[i] = np.load(covar_file)
            std_diag = np.sqrt(np.diag(self.covar_full[i]))
            npt.assert_almost_equal(std_diag, np.sqrt(self.covar_diag[start_index: end_index]))
            correlation_matrix = self.covar_full[i] / np.outer(std_diag, std_diag)
            self.covar[start_index: end_index, start_index: end_index] = correlation_matrix

    def get_covar(self, zbin=None):
        """Get the covariance matrix (full -- i.e. not the correlation matrix)"""
        if zbin is None:
            std_diag = np.sqrt(self.covar_diag)
            return self.covar * np.outer((std_diag, std_diag))
        else:
            redshift_bin_number = np.where((self.redshifts_unique < zbin + 0.1) * (self.redshifts_unique > zbin - 0.1))
            return self.covar_full[redshift_bin_number]


class HighResolutionData(BoeraData):
    """A class to store the flux power spectra and covariance matrixes for a compendium of high- (and medium-)
    resolution datasets."""
    def __init__(self):
        pass
