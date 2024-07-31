import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import LibjaxCR as jCR
import jax.numpy as jnp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from jax import grad

fs=22

start_time=time.time()

num_zeros=150
zeta_n=sp.special.jn_zeros(0,num_zeros)

R=20.0
zeta_n=jnp.array(zeta_n)
r_data=jnp.linspace(0,18,100)
gSNR_data=jCR.func_gSNR_YUK04(r_data*1.0e3)

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

ax.plot(r_data,gSNR_data,label='Original',lw=2)

N=300
theta=jnp.array([1.0e-9,2.0,1.5,4.0])
lr=0.05*theta/jnp.abs(grad(jCR.loss_func)(theta,zeta_n,R,r_data,gSNR_data))
for i in range(N):
    theta=jCR.update(theta,zeta_n,R,r_data,gSNR_data,lr)
    if((i%50==0) or (i==N-1)):
        print('i=%d -> A_fit=' %i,theta[0],'B_fit=',theta[1],'C_fit=',theta[2],'D_fit',theta[3])
        ax.plot(r_data, jCR.func_gSNR_fit(theta,zeta_n,R,r_data), label='i=%d' % i, linestyle='--')

print('A_org=',1.0/5.95828e+8,'B_org=',0.55,'C_org=',1.64,'C_org=',4.01)

ax.set_xlabel(r'$r\, ({\rm kpc})$',fontsize=fs)
ax.set_ylabel(r'$g_{\rm SNR}(r)\,({\rm pc^{-2}})$', fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.legend(loc='upper right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig('fg_gSNR.png')
plt.close()

print('Runtime:',time.time()-start_time,'seconds')

