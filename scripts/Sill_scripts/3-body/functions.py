# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:59:16 2020

@author: Sill Verberne

Code was build upon framework given in the class
"""

import numpy as np
import pars
from numba import njit

@njit
def init_3body():
    """
    construct the 3 body problem; initialize at aphelion
    """
    #declare the parameters as zeros
    xarr_ = np.zeros((2,pars.Np))#positions
    varr_ = np.zeros((2,pars.Np))#velocities
    marr_ = np.zeros(pars.Np) #masses
    #consider the SMBH (particle 0), star1 (particle 1) and star2 (particle 3)
    marr_[0] = pars.bh_mass #The SMBH
    marr_[1] = pars.star1_mass #first binary component
    marr_[2] = pars.star2_mass #second binary component

    # xarr_[:,1] = [-0.15*pars.Rin-pars.a_bin, -pars.Rin]
    # xarr_[:,2] = [-0.15*pars.Rin+pars.a_bin, -pars.Rin]
    xarr_[:,1] = [-np.sqrt(pars.Rin**2/2)+pars.a_bin, -np.sqrt(pars.Rin**2/2)]
    xarr_[:,2] = [-np.sqrt(pars.Rin**2/2)-pars.a_bin, -np.sqrt(pars.Rin**2/2)]
    vKep_bh = np.sqrt(pars.gN*pars.bh_mass/pars.Rin)

    print (vKep_bh, pars.Vk)

    #initialize with keplerian orbits for the pebble and planetesimal
    # varr_[:,1] = [vKep_bh*0.01, pars.Vk + vKep_bh*1.4]
    # varr_[:,2] = [vKep_bh*0.01,-pars.Vk + vKep_bh*1.4]
    fac = 1.6
    fac2 = 1.3
    varr_[:,1] = [fac*np.sqrt(vKep_bh**2/2), pars.Vk + fac2*np.sqrt(vKep_bh**2/2)]
    varr_[:,2] = [fac*np.sqrt(vKep_bh**2/2),-pars.Vk + fac2*np.sqrt(vKep_bh**2/2)]
    return xarr_, varr_, marr_


@njit
def forces (xarr_, varr_, marr_):
    """
    xarr_(2,Np) :positions
    marr_(Np) :masses
    Calculates the gravitational force (accelerations)
    on each particle
    returns the accelerations
    """
    acc = np.zeros((2,pars.Np))
    for value in pars.combi:
        i = value[0]
        j = value[1]
        rji = xarr_[:,j] - xarr_[:,i] #relative position (vectorial)
        r2 = np.sum((rji**2), axis=0) #squared distance (scalar)
        r1 = np.sqrt(r2) #distance
        r3 = r1*r2 #cubed distance
        force = pars.gN*rji/r3
        acc[:,i] += force*marr_[j] #add to i
        acc[:,j] -= force*marr_[i] #reverse sign
    return acc

@njit
def e_tot(xarr_, varr_, marr_):
    """
    Calculation of the total energy from the kinetic and potential energies
    """
    Ekin = np.sum(0.5*marr_*varr_**2) #Kinetic energy
    Epot = 0
    for value in pars.combi:
        i = int(value[0])
        j = int(value[1])
        rji = xarr_[:,j] - xarr_[:,i] #relative position (vectorial)
        d = np.sqrt(np.sum(rji[:]**2))
        Epot += - pars.gN * marr_[i] * marr_[j] / d #Sum over potential energy
    return Ekin + Epot

@njit
def Runge_Kutta(xarr, varr, marr_, dt):
    """
    The 4th order Runge-Kutta scheme (this is the sheme that is used)
    """
    xarr_ = xarr.copy()
    varr_ = varr.copy()
    ak1 = forces(xarr_, varr_, marr_)
    vk1 = varr_
    ak2 = forces(xarr_ + vk1*dt/2, varr_, marr_)
    vk2 = vk1 + ak1*dt/2
    ak3 = forces(xarr_ + vk2*dt/2, varr_, marr_)
    vk3 = vk1 + ak2*dt/2
    ak4 = forces(xarr_ + vk3*dt, varr_, marr_)
    vk4 = vk1 + ak3*dt
    xarr_ += (vk1 + 2*vk2 + 2*vk3 + vk4)*dt/6
    varr_ += (ak1 + 2*ak2 + 2*ak3 + ak4)*dt/6
    return xarr_, varr_
