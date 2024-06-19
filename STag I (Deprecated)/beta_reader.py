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
    
    ii_ha_beta = np.loadtxt('%s/Beta Values/ii_halpha_beta_final3.txt' % path)
    Si_beta = np.loadtxt('%s/Beta Values/ia_si_beta_final5.txt' % path)
    He_abs_beta = np.loadtxt('%s/Beta Values/he_abs_beta_final4.txt' % path)
    Ca_beta = np.loadtxt('%s/Beta Values/cal_beta_final4.txt' % path)
    fe_4924_beta = np.loadtxt('%s/Beta Values/fe_4924_beta_final2.txt' % path)
    s_beta = np.loadtxt('%s/Beta Values/s_beta_final4.txt' % path)
    he_6450_beta = np.loadtxt('%s/Beta Values/he_6450_beta_final3.txt' % path)
    fe_5018_beta = np.loadtxt('%s/Beta Values/fe_5018_beta_final2.txt' % path)
    fe_5170_beta = np.loadtxt('%s/Beta Values/fe_5170_beta_final2.txt' % path)
    ii_hg_beta = np.loadtxt('%s/Beta Values/ii_hg_beta_final2.txt' % path)
    si_4000_beta = np.loadtxt('%s/Beta Values/si_4000_beta_final.txt' % path)
    hbeta_beta = np.loadtxt('%s/Beta Values/hbeta_beta_final.txt' % path)

    return ii_ha_beta,Si_beta,He_abs_beta,Ca_beta,fe_4924_beta,s_beta,he_6450_beta,fe_5018_beta,fe_5170_beta,ii_hg_beta,si_4000_beta,hbeta_beta