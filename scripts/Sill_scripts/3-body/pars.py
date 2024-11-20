# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:01:16 2020

@author: Sill Verberne
"""

import numpy as np
from itertools import combinations #Used to get rid of the nested for loops

#file pars.py
#constants (CGS)
gN = 6.67408e-08 #Newton’s gravitational constant
mSun = 1.9884754153381438e+33 #mass of the Sun
au = 1.495978707e13 #astronomical unit (in cm)
yr = 2*np.pi /np.sqrt(gN*mSun/au**3) #1 year in seconds

star1_mass = 2 * mSun # Mass of star 1 in binary
star2_mass = 2 * mSun # Mass of star 2 in binary
a_bin = 0.1 * au # Binary separation
Rin = 1000 * au # Initial distance between binary and SMBH
bh_mass = 4*10**6 * mSun # Mass of the SMBH

P = np.sqrt(((2*a_bin)**3 * 4 * np.pi**2)/(gN*(star1_mass+star2_mass)))
Vk = 2*np.pi*a_bin/P

#problem parameters
Np = 3 #3 particles

#possible combinations between all the objects
particles = [i for i in range(Np)]
combi = np.array(list(combinations(particles, 2))) #All combinations
