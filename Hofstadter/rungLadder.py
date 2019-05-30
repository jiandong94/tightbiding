#!/usr/bin/env python

# one dimensional tight-binding model of a trestle-like structure

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
import sys
path = '/home/chenjd/Code/TightBiding/src'
sys.path.append(path)
from tbpy import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt
#  --- --- --- --- --- --- ---
# |   |   |   |   |   |   |   |
#  --- --- --- --- --- --- ---
#         2D ladder

p = 1.0
q = 2.0
phi = p/q*np.pi
nq = int(q)*2
norb = nq*2
# define lattice vectors
lat=[[1.0,0.0],[0.0,1.0]]
# define coordinates of orbitals
orb = []
for i in range(nq):
    orb.append([1.0/q*i, 0.0])
    orb.append([1.0/q*i, 1.0])

# make one dimensional tight-binding model of a trestle-like structure
ladder = tb_model(1,2,lat,orb,per=[0],nspin=1)

# set model parameters
#t_para = np.zeros((2,2), dtype=complex)
#t_vert = np.zeros((2,2), dtype=complex)
#t_para[0,0] = -1.0
#t_para[1,1] = -1.0
t_para = -1.
for i in range(nq):
    #t_vert[0,0] = -1.0*np.exp(-1j*(nq+1)*phi)
    #t_vert[1,1] = -1.0*np.exp(-1j*(nq+1)*phi)
    t_vert = -1.0*np.exp(-1j*(i+1)*phi)
    print(i,t_vert)
    site = 2*i
    if i == nq-1:
        ladder.set_hop(t_para, site, 0, [1,0])
        ladder.set_hop(t_para, site+1, 1, [1,0])
        ladder.set_hop(t_vert, site, site+1, [0,0])
    else:
        ladder.set_hop(t_para, site, site+2, [0,0])
        ladder.set_hop(t_para, site+1, site+3, [0,0])
        ladder.set_hop(t_vert, site, site+1, [0,0])


ladder.display()
# generate list of k-points following some high-symmetry line in
path = [[-0.5],[0.],[0.5]]
(k_vec,k_dist,k_node) = ladder.k_path(path, 1000)
k_label=[r"$-\pi$",r"$0$", r"$\pi$"]

print('---------------------------------------')
print('starting calculation')
print('---------------------------------------')
print('Calculating bands...')

# solve for eigenenergies of hamiltonian on
# the set of k-points from above
evals = ladder.solve_all(k_vec)

# plotting of band structure
print('Plotting bandstructure...')

# First make a figure object
fig, ax = plt.subplots()
# specify horizontal axis details
ax.set_xlim(k_node[0],k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(k_label)
ax.axvline(x=k_node[1],linewidth=0.5, color='k')

# plot band
ax.plot(k_dist,evals[0])
print(k_dist[-1])
print(k_dist[np.where(evals[0]==np.min(evals[0]))])
ax.plot(k_dist,evals[1])
#for i in range(ladder.get_nsta()):
#    ax.plot(k_dist,evals[i])
# put title
#ax.set_title("Band structure")
ax.set_xlabel("k")
ax.set_ylabel("Band energy")
plt.show()
# make an PDF figure of a plot
#fig.tight_layout()
#fig.savefig("trestle_band.pdf")

print('Done.\n')
