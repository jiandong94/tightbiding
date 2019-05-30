#!/usr/bin/env python
# Toy graphene model
from __future__ import print_function
from tbpy import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# define lattice vectors
lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
# define coordinates of orbitals
orb=[[1./3.,1./3.],[2./3.,2./3.]]
# make two dimensional tight-binding graphene model
my_model=tb_model(2,2,lat,orb)
# set model parameters
delta=0.0
t=-1.0
# set on-site energies
my_model.set_onsite([-delta,delta])
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [ 0, 0])
my_model.set_hop(t, 1, 0, [ 1, 0])
my_model.set_hop(t, 1, 0, [ 0, 1])
# generate list of k-points following a segmented path in the BZ
path=[[0.,0.],[2./3.,1./3.],[.5,.5],[0.,0.]]
label=(r'$\Gamma $',r'$K$', r'$M$', r'$\Gamma $')
nk=121
(k_vec,k_dist,d_node)=my_model.k_path(path,nk)
evals=my_model.solve_all(k_vec)
# figure for bandstructure
fig, ax = plt.subplots()
ax.set_xlim(d_node[0],d_node[-1])
ax.set_xticks(d_node)
ax.set_xticklabels(label)
for n in range(len(d_node)):
  ax.axvline(x=d_node[n],linewidth=0.5, color='k')
ax.set_title("Graphene band structure")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy")
# plot first and second band
ax.plot(k_dist,evals[0])
ax.plot(k_dist,evals[1])

### 2D BZ
fig = plt.figure()
ax = Axes3D(fig)
#x = np.linspace(-2.0/3.0, 2.0/3.0, 100)
#y = np.linspace(-1.0/2.0, 1.0/2.0, 100)
x = np.linspace(-1, 1, 150)
y = np.linspace(-2.0/3.0, 2.0/3.0, 150)
[X, Y] = np.meshgrid(x,y)
krow = np.shape(X)[0]
kcol = np.shape(X)[1]
Z = np.zeros((my_model._nsta,krow,kcol), dtype=float)
for i in range(krow):
    for j in range(kcol):
        ham = my_model._gen_ham([X[i,j], Y[i,j]])
        eval = my_model._sol_ham(ham, eig_vectors=False)
        Z[:,i,j] = eval
ax.plot_surface(X, Y, Z[0,:,:], rstride=1, cstride=1)
ax.plot_surface(X, Y, Z[1,:,:], rstride=1, cstride=1)
ax.set_xlabel('x label', color='r')
ax.set_ylabel('y label', color='g')
ax.set_zlabel('z label', color='b')



plt.show()
#fig.savefig("graphene.pdf")

print('Done.\n')
