import numpy as np
import astropy as ap
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt
import bayesian_LC as blc
from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import emcee
import corner

def prior_sample(n):
    return np.array([[np.random.uniform(low=-17,high=-12),np.random.uniform(low=0,high=1)] for i in range(n)])

kne_inf = blc.Kilonova_Inference()

m_low = 15
m_high = 27
mlims_f_j = np.array([[22,26,26,26],[21,27,27,27]])
f_bar_sum = kne_inf.compute_f_bar_sum(mlims_f_j)
t0 = 1*u.day
T_f_j = [(np.linspace(t0.value,t0.value+4,4))*u.day,(np.linspace(t0.value,t0.value+4,4))*u.day]
P_f = np.array([0.999,0.])
p_d_f = np.array([kne_inf.create_distance_dist(156,41),kne_inf.create_distance_dist(156,41)])
P_A= 0.2
P_T = 1.-P_A

m_low_arr = np.ones_like(mlims_f_j)*m_low
m_high_arr = np.ones_like(mlims_f_j)*m_high
total_obs = sum(len(m) for m in mlims_f_j)
plims_f_bar = 1/(m_high-m_low)**(total_obs)*(1-sum(P_f))
plims_T = 1/(m_high-m_low)**(total_obs)

theta = np.array([[np.random.uniform(low=-17,high=-12),np.random.uniform(low=0,high=1)] for i in range(100)])
y = np.array([kne_inf.ln_prob(th,mlims_f_j,T_f_j,t0,p_d_f,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr) for th in theta])
np.savetxt('theta.txt',theta)
np.savetxt('y.txt',y)
y_no_inf = np.nan_to_num(y,neginf=-600.)
y_no_inf[y_no_inf<=-600.] = -600.
#theta = np.loadtxt('theta.txt')
#y_no_inf = np.nan_to_num(np.loadtxt('y.txt'),neginf=-710.)
gp = gpUtils.defaultGP(theta, y_no_inf, white_noise=-12)

m=200
nmax=3
bounds = [(-17,-12),(0,1)]
algorithm = 'bape'
samplerKwargs = {"nwalkers" : 20}
mcmcKwargs = {"iterations" : 8000}

ap = approx.ApproxPosterior(theta=theta, y=y_no_inf, gp=gp, lnprior=kne_inf.ln_prior, lnlike=kne_inf.ln_likelihood,
                            priorSample=prior_sample, algorithm=algorithm, bounds=bounds)

ap.run(m=m, convergenceCheck=True, estBurnin=True, nGPRestarts=3, mcmcKwargs=mcmcKwargs,
       cache=False, samplerKwargs=samplerKwargs, verbose=True, thinChains=True,
       optGPEveryN=5,args=[mlims_f_j,T_f_j,t0,p_d_f,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr])

samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    range=[(-17,-12),(0,1)],scale_hist=True, plot_contours=True,labels=[r"$M0$",r"$\gamma$"],title_kwargs={"fontsize": 14})
plt.savefig(fname='../plots/uniform-likelihood-simplified-2f-1-0-PA0p2-22-26-26-26-and-21-27-27-27.png',format='png')




