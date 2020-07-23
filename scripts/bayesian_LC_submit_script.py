import bayesian_LC as blc
import numpy as np
import astropy.units as u

kne_inf = blc.Kilonova_Inference()

m_low = 15
m_high = 27
mlims_f_j = np.array([[22,26,26,26],[19,22,22,22]])
f_bar_sum = kne_inf.compute_f_bar_sum(mlims_f_j)
t0 = 1*u.day
T_f_j = [(np.linspace(t0.value,t0.value+4,4))*u.day,(np.linspace(t0.value,t0.value+4,4))*u.day]
P_f = np.array([0.,1.])
p_d_f = np.array([kne_inf.create_distance_dist(156,41),kne_inf.create_distance_dist(156,41)])
P_A= 1.
P_T = 1.-P_A

m_low_arr = np.ones_like(mlims_f_j)*m_low
m_high_arr = np.ones_like(mlims_f_j)*m_high
total_obs = sum(len(m) for m in mlims_f_j)
plims_f_bar = 1/(m_high-m_low)**(total_obs)*(1-sum(P_f))
plims_T = 1/(m_high-m_low)**(total_obs)


sampler = blc.Sampler(2,20,[[-17,-12],[0,1]],mlims_f_j,T_f_j,t0,p_d_f,P_f,P_A,P_T,f_bar_sum,plims_f_bar,plims_T,m_low_arr,m_high_arr)

samples = sampler.sample(nburn=1000,nsteps=1000)

fig = sampler.corner_plot(samples,fname='../plots/uniform-likelihood-simplified-2f-0p05-0p95-PA1-22-26-26-26-and-19-22-22-22.png')


