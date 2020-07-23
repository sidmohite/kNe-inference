import numpy as np
import astropy as ap
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt
from scipy.stats import norm,multivariate_normal
from scipy.integrate import quad
from scipy.stats import truncnorm
import emcee
from multiprocessing import Pool
import corner
import healpy as hp
import time

class Skymap_Probability():
    def __init__(self,skymap_fits_file,field_coords_array):
        self.skymap_file = skymap_fits_file
        self.field_coords_array = field_coords_array
        prob,distmu,distsigma,distnorm = hp.read_map(self.skymap_file,field=range(4))
        npix = len(prob)
        self.nside = hp.npix2nside(npix)
        self.prob = prob
        self.distmu = distmu
        self.distsigma = distsigma
        self.distnorm = distnorm

    def pix_in_fields(self):
        return np.array([hp.query_polygon(self.nside,coords) for coords in self.field_coords_array]) 
    
    def calculate_field_probs(self):
        ipix_fields = self.pix_in_fields()
        return np.array([self.prob[ipix] for ipix in ipix_fields])
    
    def construct_margdist_distribution(self,ipix_field,field_prob):
        dp_dr = lambda r:np.sum(self.prob[ipix_field]*r**2*self.distnorm[ipix_field]*
                                norm(self.distmu[ipix_field],self.distsigma[ipix_field]).pdf(r))/np.sum(field_prob)
        return dp_dr

class Kilonova_Inference():
    def __init__(self):
        print("Initializing inference framework...")
    
    def lc_model(self,M_0,gamma,t_0,t):
        return (M_0*pow(t_0/t,gamma))*u.mag

    def M_to_m(self,M,distance):
        return (M + 5*np.log10(distance*1e6) -5)

    def dlim(self,mlim,M_j):
        #Answer in Mpc
        return 10**((mlim-M_j)/5.)*10*1e-6

    def create_distance_dist(self,mu_f,sigma_f):
        a = (0-mu_f)/sigma_f
        b = (4000.-mu_f)/sigma_f
        return truncnorm(a,b,mu_f,sigma_f)

    def create_mlim_pdf(self,m_lim,M,p_d,m_low,m_high):
        num = quad(p_d.pdf,self.dlim(m_lim,M),np.inf)[0]
        den = quad(lambda m: quad(p_d.pdf,self.dlim(m,M),np.inf)[0],m_low,m_high)[0]
        return num/den

    def map_func(self,func,vals):
        return np.array(list(map(func,vals)))

    def compute_f_bar_sum(self,mlims_f_j):
        total_len = sum(len(m) for m in mlims_f_j)
        return np.array([total_len -len(mlims) for mlims in mlims_f_j])

    def ln_prior(self,params):
        M0,gamma = params
        if (-17<=M0<=-12)&(0<=gamma<=1):
            return 0.0
        return -np.inf

    def ln_likelihood(self,params,mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr):
        M0,gamma = params
        M_f_j = np.array([self.lc_model(M0,gamma,t_0=t0,t=t_j).value for t_j in T_f_j])
        dlims_f_j = np.array(list(map(self.dlim,mlims_f_j,M_f_j)))
        pmlims_f_j = np.array(list(map(np.vectorize(self.create_mlim_pdf),mlims_f_j,M_f_j,p_d,m_low_arr,m_high_arr)))
        pmlims_f = np.array([np.product(p) for p in pmlims_f_j])*P_f/(m_high_arr[0][0]-m_low_arr[0][0])**f_bar_sum
        return np.sum(np.log((np.sum(pmlims_f) + plims_f_bar)*P_A + plims_T*P_T))

    def ln_likelihood_events(self,params,mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,total_obs,m_low,m_high):
        ln_likelihood_arr = np.array(list(map(self.ln_likelihood,params,mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,total_obs,m_low,m_high)))
        return np.sum(ln_likelihood_arr)

    def ln_prob(self,params,mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr):
#        lnp = self.ln_prior(params)
        M0,gamma = params
        if (-17<=M0<=-12)&(0<=gamma<=1):
            return self.ln_likelihood(params,mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr)
        return -np.inf

class Sampler(Kilonova_Inference):
    def __init__(self,ndim,nwalkers,limits,mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr):
        Kilonova_Inference.__init__(self)
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.limits = limits
        self.data = (mlims_f_j,T_f_j,t0,p_d,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr)

    def sample(self,nburn,nsteps,pool=False):
        pos0 = [[np.random.uniform(low=self.limits[0][0],high=self.limits[0][1]),np.random.uniform(low=self.limits[1][0],high=self.limits[1][1])] for i in range(self.nwalkers)]
        if pool:
            pool_inst = Pool()
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.ln_prob, args=self.data, pool=pool_inst)
            pos, prob, state = sampler.run_mcmc(pos0, nburn,progress=True)
            sampler.reset()
            sampler.run_mcmc(pos,nsteps,progress=True)
            return sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.ln_prob, args=self.data)
        pos, prob, state = sampler.run_mcmc(pos0, nburn,progress=True)
        sampler.reset()
        sampler.run_mcmc(pos,nsteps,progress=True)
        return sampler
        
    def corner_plot(self,sampler,fname):
        fig = corner.corner(sampler.flatchain,labels=[r"$M0$",r"$\gamma$"],
              quantiles=[0.02, 0.5, 0.98],
              show_titles=True, title_kwargs={"fontsize": 14})
        plt.savefig(fname=fname,format='png')

