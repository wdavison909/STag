#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:48:51 2021

@author: willdavison
"""

import numpy as np

def beta_reader():
    """
    Function to read in beta values for each tag
    """
    H_beta = np.loadtxt('/Beta Values/h_beta_final2.txt')
    Si_beta = np.loadtxt('/Beta Values/si_beta_final2.txt')
    He_emi_beta = np.loadtxt('/Beta Values/he_emi_beta_final2.txt')
    He_cyg_beta = np.loadtxt('/Beta Values/he_cyg_beta_final2.txt')
    He_abs_beta = np.loadtxt('/Beta Values/he_abs_beta_final2.txt')
    H_alp_beta = np.loadtxt('/Beta Values/h_cyg_beta_final2.txt')
    Ca_beta = np.loadtxt('/Beta Values/ca_beta_final2.txt')
    iib_dp_beta = np.loadtxt('/Beta Values/iibdp_beta_final2.txt')
    Fe_beta = np.loadtxt('/Beta Values/5000A_beta_final2.txt')
    S_beta = np.loadtxt('/Beta Values/5600A_beta_final2.txt')

    return H_beta,Si_beta,He_emi_beta,He_cyg_beta,He_abs_beta,H_alp_beta,Ca_beta,iib_dp_beta,Fe_beta,S_beta