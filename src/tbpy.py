from __future__ import print_function
# Learn from PythTB

import numpy as np
import sys
import copy
class tb_model(object):
    r"""Get the band struct with tight-binding model

    Parameters
    ----------
    dim_k : int
        Dimension of reciprocal space, which specifies how many directions are
        considered to be periodic.
    dim_r : int
        Dimension of real space, which specifies how many real space vectors are
        needed to get the coordinates of sites.
    lat : array of floats
        Lattice vectors in Cartesian coordinates, eg,[[1.0,0.0],[0.0,1.0]] in 
        2D situation.
    orb : array of floats
        Orbtial coordinates (sites coordinates in unit cell) in reduced coordinates.
    per : list of ints
        Indices which specifies which lattice vectors considered to be periodic. By
        defalt, all directions are periodic.
    nspin : int
        Number of spin components for each orbitals. If *nspin* is 1, then the model 
        is spinless, if *nspin* is 2, then it is spinful. By defalt, this parameter is 
        1.

    Attributes
    ----------
    _dim_k
    _dim_r
    _lat
    _orb
    _per
    _nspin
    _norb : int
        The number of orbitals
    _nsta : int
        Number of states for each k-point, which equals *orb\*nspin*.
    _onsite_en : array of floats 
        Onsite energy for each orbital, by defalt, it is zero.
    _onsite_en_spec : array of bools
        An array of bools to specify  
    _hopping : list
    _hopping_spec  : bool
    """
    def __init__(self,dim_k,dim_r,lat=None,orb=None,per=None,nspin=1):
        self._dim_k = dim_k
        self._dim_r = dim_r
        self._lat = np.array(lat, dtype=float)
        self._orb = np.array(orb, dtype=float)
        self._norb = self._orb.shape[0]
        self._nspin = nspin
        
        # choose periodic directions
        if per == None:
            self._per = list(range(self._dim_k))
        else:
            self._per = per


        # number of state for each k-point
        self._nsta = self._norb*self._nspin

        # initialize onsite energy zeros
        if nspin == 1:
            self._onsite_en = np.zeros((self._norb), dtype=float)
        elif self._nspin == 2:
            self._onsite_en = np.zeros((self._norb,2,2), dtype=complex)

        # initialize onsite energy unspecified
        self._onsite_en_spec = np.zeros((self._norb), dtype=bool)
        self._onsite_en_spec[:] = False

        # initialize hopping term to list
        self._hopping = []
        self._hopping_spec = False

    @classmethod
    def set_onsite(self,onsite_en,ind_i=None,mode="set"):
        r"""Define on-site energy for tight-biding models.
        
        Parameters
        ----------
        onsite_en : array of floats
            A list of energy for each orbitals.
        ind_i : int
            Index of obtial whose on-site energy you wish to change.
        mode : string
            Specify the way you wish to change on-site energy.

            * "set" Defalt value.
        
        """
        # check len(onsite_en) == len(self_orb) 
        if ind_i == None:
            if (len(onsite_en) != self._norb):
                raise Exception("\n\ndim of onsite_en is not equal to orb's")
        
        # check ind_i is not out of range
        if ind_i != None:
            if ind_i < 0 or ind_i >= self._norb:
                raise Exception("\n\nind_i is out of range")

        # choose mode: "set","reset","add"
        if mode.lower() == "set":
            if ind_i != None:
                if self._onsite_en_spec[ind_i] == True:
                    raise Exception("\n\nonsite energy at this site was specified!")
                else:
                    self._onsite_en[ind_i] = self._val_to_block(onsite_en)
                    self._onsite_en_spec[ind_i] = True
            else:
                if True in self._onsite_en_spec:
                    raise Exception("\n\nonsite energy at some sites were specified!")
                else:
                    for i in range(self._norb):
                        self._onsite_en[i] = self._val_to_block(onsite_en[i])
                    self._onsite_en_spec[:] = True
    
    def set_hop(self,hop_amp,ind_i,ind_j,ind_R=None,mode="set"):
        
        # check ind_i is not out of range
        if ind_i < 0 or ind_i >= self._norb:
            raise Exception("\n\nind_i is out of range")
        if ind_j < 0 or ind_j >= self._norb:
            raise Exception("\n\nind_j is out of range")

        # check onsite energy is not counted in set_hop
        if self._dim_k == 0:
            if ind_i == ind_j:
                raise Exception("\n\n Onsite energy is counted here!")
        else:
            if ind_i == ind_j:
                onsite_check = True
                for k in self._per:
                    if int(ind_R[k]) != 0:
                        onsite_check = False
                if onsite_check == True:
                    raise Exception("\n\n Onsite energy is counted here!")
        
        # consider spin
        hop_amp = self._val_to_block(hop_amp)
        # hopping parameters to be stored
        if self._dim_k == 0:
            hop_para = [hop_amp, int(ind_i), int(ind_j)]
        else:
            hop_para = [hop_amp, int(ind_i), int(ind_j), np.array(ind_R)]


        # choose mode: "set","add"
        if mode.lower() == "set":
            if self._hopping_spec == True:
                raise Exception("\n\n This hopping term is repeated")
            else:
                self._hopping.append(hop_para)
        
        
    def _val_to_block(self,val):
        # spinless
        if self._nspin == 1:
            return val
        # spinful
        elif self._nspin == 2:
            val_mat = np.zeros((2,2), dtype=complex)
            val = np.array(val)
            if val.shape == ():
                val_mat[0,0] = val
                val_mat[1,1] = val
            elif val.shape == (2,2):
                return val
            return val_mat

    def get_nsta(self):
        return self._nsta
            
    def _gen_ham(self,k_input=None):
        kpoint = np.array(k_input)
        if kpoint is not None:
            # kpoint is a single number
            if kpoint.shape == ():
                kpoint = np.array([kpoint])
            # check the size
            if kpoint.shape != (self._dim_k,):
                raise Exception("\n\n The shape of k_input is wrong")
        else:
            raise Exception("\n\n Please input k_input")

        # Initial Hamiltonian matrix
        # spinless
        if self._nspin == 1:
            ham = np.zeros((self._norb, self._norb), dtype=complex)
        elif self._nspin == 2:
            ham = np.zeros((self._norb,2,self._norb,2), dtype=complex)

        # Onsite energy
        for i in range(self._norb):
            if self._nspin == 1:
                ham[i,i] = self._onsite_en[i]
            elif self._nspin == 2:
                ham[i,:,i,:] = self._onsite_en[i]

        # Hopping
        for hopping in self._hopping:
            amp = hopping[0]
            i = hopping[1]
            j = hopping[2]
            # 0 dim
            if self._dim_k > 0:
                ind_R = np.array(hopping[3], dtype=float)
                # calculate R_{j} - R_{i}
                deltaR = self._orb[j,:] + ind_R - self._orb[i,:]
                # rake periodic components
                deltaR = deltaR[self._per]
                # calculate phase factor
                phase = np.exp((2.0j)*np.pi*np.dot(deltaR, kpoint))
                amp = amp*phase
            if self._nspin == 1:
                ham[i,j] += amp
                ham[j,i] += amp.conjugate()
            elif self._nspin == 2:
                ham[i,:,j,:] += amp
                ham[j,:,i,:] += amp.T.conjugate()
        return ham

    def _sol_ham(self,ham,eig_vectors=False):
        # reshape Hamiltonian matrix
        if self._nspin == 1:
            ham_use = ham;
        elif self._nspin == 2:
            ham_use = np.reshape(ham, (2*self._norb, 2*self._norb))

        # check hermitian
        if np.max(np.abs(ham_use-ham_use.T.conj())) > 1.0E-9:
            raise Exception("\n\n Hamiltonian is not hermitian")

        if eig_vectors == False:
            eval = np.linalg.eigvalsh(ham_use)
            eval = _nice_eig(eval)
            return np.array(eval, dtype=float)
        else:
            (eval,evec) = np.linalg.eigh(ham_use)
            # transform to evec[i,:]
            evec = evec.T
            # sort eigenvalues and eigenvectors
            (eval,evec) = _nice_eig(eval,evec)

            if self._nspin == 2:
                evec = np.reshape(evec,(self._nsta,self._norb,2))  #### ?
            return (eval,evec)

    def solve_all(self,k_list=None,eig_vectors=False):
        # 0-dim case
        if k_list is None:
            if self._dim_k != 0:
                raise Exception("\n\n k_list can not be None if dim != 0")
            ham = self._gen_ham()
            if eig_vectors == False:
                eval = self._sol_ham(ham,eig_vector=eig_vector)
                return eval
            else:
                (eval,evec) = self._sol_ham(ham,eig_vector=eig_vector)
                return (eval, evec)
        # an array of k_list
        else:
            # number of k-points
            nkpoint = len(k_list)
            # return data
            # (band, kpoint)
            return_eval = np.zeros((self._nsta, nkpoint), dtype=float)
            # (band, kponit, orbital, spin)
            if self._nspin == 1:
                return_evec = np.zeros((self._nsta, nkpoint, self._norb), dtype=complex)
            elif self._nspin == 2:
                return_evec = np.zeros((self._nsta, nkpoint, self._norb, self._nspin), dtype=complex)

            # loop for every k-point
            for i,k in enumerate(k_list):
                ham = self._gen_ham(k)
                if eig_vectors == False:
                    eval = self._sol_ham(ham, eig_vectors=eig_vectors)
                    return_eval[:,i] = eval[:]
                else: 
                    (eval,evec) = self._sol_ham(ham, eig_vectors=eig_vectors)
                    return_eval[:,i] = eval[:]
                    if self._nspin == 1:
                        return_evec[:,i,:] = evec[:,:]
                    elif self._nspin == 2:
                        return_evec[:,i,:,:] = evec[:,:,:]

            if eig_vectors == False:
                return return_eval
            else:
                return (return_eval,return_evec)
    
    def solve_one(self,k_point=None,eig_vectors=False):
        if k_point == None:
            return self.solve_all(eig_vectors=eig_vectors)
        else:
            if eig_vectors == False:
                eval = self.solve_all([k_point],eig_vectors=eig_vectors)
                return eval[:,0]
            else:
                (eval,evec) = self.solve_all([k_point],eig_vectors=eig_vectors)
                if self._nspin == 1:
                    return (eval[:,0], evec[:,0,:])
                elif self._nspin == 2:
                    return (eval[:,0], evec[:,0,:,:])

    def k_path(self,hspts,nk,report=True):
        # generate path of high-symmetry points in nk points
        # (n_nodes, dim_k)
        k_list = np.array(hspts, dtype=float)
        # 1D case eg: hspts=[1,2,3]
        if len(k_list.shape) == 1 and self._dim_k == 1:
            k_list = np.array([k_list]).T
        
        # number of nodes
        n_nodes = k_list.shape[0]

        # distance of nodes
        d_nodes = np.zeros(n_nodes, dtype=float)
        # a copy of lattice vector choosing periodic directions
        lat_per = np.copy(self._lat)
        lat_per = lat_per[self._per]
        # k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n-1]
            dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
            d_nodes[n] = d_nodes[n-1] + dklen
        
        # number of k-point of nodes
        nk_nodes = [0]
        for n in range(1, n_nodes-1):
            frac = d_nodes[n]/d_nodes[-1]
            nk_nodes.append(int(round(frac*(nk-1))))
        nk_nodes.append(nk-1)

        # k-distance and k-points
        k_dist = np.zeros(nk, dtype=float)
        k_vec = np.zeros((nk, self._dim_k), dtype=float)

        # loop for all k-points
        k_vec[0] = k_list[0]
        for n in range(1, n_nodes):
            nk_i = nk_nodes[n-1]
            nk_j = nk_nodes[n]
            d_i = d_nodes[n-1]
            d_j = d_nodes[n]
            k_i = k_list[n-1]
            k_j = k_list[n]
            for m in range(nk_i, nk_j+1):
                frac = float(m-nk_i)/float(nk_j-nk_i)
                k_dist[m] = d_i + frac*(d_j-d_i)
                k_vec[m] = k_i + frac*(k_j-k_i)
        return (k_vec,k_dist,d_nodes)
        

    def display(self):
        print("==============================")
        print("k-space dimension          =", self._dim_k)
        print("r-space dimension          =", self._dim_r)
        print("number of orbitals         =", self._norb)
        print("number of spin components  =", self._nspin)
        print("periodic directions        =", self._per)
        print("number of states           =", self._nsta)

        print("hopping term:")
        for i,hopping in enumerate(self._hopping):
            print("<",hopping[1],"|H|",hopping[2],"+",hopping[3],">   =",_nice_type(hopping[0],2))
def _nice_eig(eval,evec=None):
    # sort eigenvalues(real) and eigenvectors
    eval = np.array(eval.real, dtype=float)
    args = eval.argsort()
    eval = eval[args]
    if evec == None:
        return eval
    else:
        evec = evec[args]
        return (eval, evec)

def _nice_type(num,k=2):
    if type(num) == int or type(num) == float:
        return round(num, k)
    elif type(num) == complex or type(num) == np.complex128:
        return round(num.real, k)+round(num.imag, k)*1j
    else:
        return num






