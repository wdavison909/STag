#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:48:51 2021

@author: willdavison
"""

import numpy as np

def beta_reader(direc):
    """
    Function to read in beta values for each tag
    """
    path = direc
    H_beta = np.loadtxt('%s/Beta Values/h_beta_final2.txt' % path)
    Si_beta = np.loadtxt('%s/Beta Values/si_beta_final2.txt' % path)
    He_emi_beta = np.loadtxt('%s/Beta Values/he_emi_beta_final2.txt' % path)
    He_cyg_beta = np.loadtxt('%s/Beta Values/he_cyg_beta_final2.txt' % path)
    He_abs_beta = np.loadtxt('%s/Beta Values/he_abs_beta_final2.txt' % path)
    H_alp_beta = np.loadtxt('%s/Beta Values/h_alp_beta_final2.txt' % path)
    Ca_beta = np.loadtxt('%s/Beta Values/ca_beta_final2.txt' % path)
    iib_dp_beta = np.loadtxt('%s/Beta Values/iibdp_beta_final2.txt' % path)
    Fe_beta = np.loadtxt('%s/Beta Values/fe_beta_final2.txt' % path)
    S_beta = np.loadtxt('%s/Beta Values/s_beta_final2.txt' % path)

    return H_beta,Si_beta,He_emi_beta,He_cyg_beta,He_abs_beta,H_alp_beta,Ca_beta,iib_dp_beta,Fe_beta,S_beta