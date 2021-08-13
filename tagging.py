#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:12:57 2021

@author: willdavison
"""

import numpy as np

def getType(tp, subtp):
    """
    Originally from https://github.com/fedhere/SESNspectraPCA
    
    Convert tuple type designation from SNID to string.
    Parameters
    ----------
    tp : int
        SNID type int from template
    subtp : int
        SNID subtype int from template
    Returns
    -------
    sntype : string
    snsubtype : string
    """
    if tp == 1:
        sntype = 'Ia'
        if subtp == 2: 
            snsubtype = 'norm'
        elif subtp == 3: 
            snsubtype = '91T'    
        elif subtp == 4: 
            snsubtype = '91bg'
        elif subtp == 5: 
            snsubtype = 'csm'
        elif subtp == 6: 
            snsubtype = 'pec'
        elif subtp == 7:
            snsubtype = '99aa'
        elif subtp == 8: 
            snsubtype = 'Iax'
        else: 
            snsubtype = ''
    elif tp == 2:
        sntype = 'Ib'
        if subtp == 2: 
            snsubtype = 'norm'
        elif subtp == 3: 
            snsubtype = 'pec'
        elif subtp == 4:
            snsubtype = 'IIb'
        elif subtp == 5: 
            snsubtype = 'Ibn'
        elif subtp == 6:
            snsubtype = 'Ca'
        else: 
            snsubtype = ''
    elif tp == 3:
        sntype = 'Ic'
        if subtp == 2: 
            snsubtype = 'norm'
        elif subtp == 3: 
            snsubtype = 'pec'
        elif subtp == 4:
            snsubtype = 'broad'
        elif subtp == 5:
            snsubtype = 'SL'
        else: 
            snsubtype = ''
    elif tp == 4:
        sntype = 'II'
        if subtp == 2: 
            snsubtype = 'P'
        elif subtp == 3: 
            snsubtype = 'pec'
        elif subtp == 4: 
            snsubtype = 'n'
        elif subtp == 5: 
            snsubtype = 'L'
        else: 
            snsubtype = ''
    return sntype, snsubtype

class SNIDsn:
    """
    Originally from https://github.com/fedhere/SESNspectraPCA
    """
    def __init__(self):
        self.header = None
        self.continuum = None
        self.phases = None
        self.phaseType = None
        self.wavelengths = None
        self.data = None
        self.type = None
        self.subtype = None

        self.smoothinfo = dict()
        self.smooth_uncertainty = dict()

        return

    def loadSNIDlnw(self, lnwfile):
        """
        Loads the .lnw SNID template file specified by the path lnwfile into
        a SNIDsn object.
        Parameters
        ----------
        lnwfile : string
            path to SNID template file produced by logwave.
        Returns
        -------
        """
        with open(lnwfile) as lnw:
            lines = lnw.readlines()
            lnw.close()
        header_line = lines[0].strip()
        header_items = header_line.split()
        header = dict()
        header['Nspec'] = int(header_items[0])
        header['Nbins'] = int(header_items[1])
        header['WvlStart'] = float(header_items[2])
        header['WvlEnd'] = float(header_items[3])
        header['SplineKnots'] = int(header_items[4])
        header['SN'] = header_items[5]
        header['dm15'] = float(header_items[6])
        header['TypeStr'] = header_items[7]
        header['TypeInt'] = int(header_items[8])
        header['SubTypeInt'] = int(header_items[9])
        self.header = header

        tp, subtp = getType(header['TypeInt'], header['SubTypeInt'])
        self.type = tp
        self.subtype = subtp

        phase_line_ind = len(lines) - self.header['Nbins'] - 1
        phase_items = lines[phase_line_ind].strip().split()
        self.phaseType = int(phase_items[0])
        phases = np.array([float(ph) for ph in phase_items[1:]])
        self.phases = phases

        wvl = np.loadtxt(lnwfile, skiprows=phase_line_ind + 1, usecols=0)
        self.wavelengths = wvl
        lnwdtype = []
        colnames = []
        for ph in self.phases:
            colname = 'Ph'+str(ph)
            if colname in colnames:
                colname = colname + 'v1'
            count = 2
            while(colname in colnames):
                colname = colname[0:-2] + 'v'+str(count)
                count = count + 1
            colnames.append(colname)
            dt = (colname, 'f4')
            lnwdtype.append(dt)
        #lnwdtype = [('Ph'+str(ph), 'f4') for ph in self.phases]
        #print lines[phase_line_ind+1]
        data = np.loadtxt(lnwfile, dtype=lnwdtype, skiprows=phase_line_ind + 1, usecols=range(1,len(self.phases) + 1))
        self.data = data

        continuumcols = len(lines[1].strip().split())
        continuum = np.ndarray((phase_line_ind - 1,continuumcols))
        for ind in np.arange(1,phase_line_ind - 0):
            cont_line = lines[ind].strip().split()
            #print cont_line
            continuum[ind - 1] = np.array([float(x) for x in cont_line])
        self.continuum = continuum
        return

def unison_shuffled_copies(a, b):
    """
    Description
    -----------
    Function to shuffle multiple arrays in the same way
    i.e. if the first row of 'a' is shuffled to the fifth
    row, the same is done for 'b'.
    
    Parameters
    ----------
    a: an array
    b: an array

    Returns
    -------
    a[p]: shuffled array 'a', permutation 'p'
    b[p]: shuffled array 'b', permutation 'p'
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def param_sum(x,beta):
  """
  Description
  -----------
  Sum the beta parameters of the logistic function.

  Parameters
  ----------
  x: flux values
  beta: beta values

  Returns
  -------
  sum_p: summation of beta values multiplied by the flux values
  """
  sum_p = beta[0]
  for i in range(len(x)):
    sum_p += beta[i+1]*x[i]

  return sum_p

def log_reg(sum_p):
  """
  Description
  -----------
  Calculate the value of the logistic regression function for
  a specfic combination of beta values.

  Parameters
  ----------
  sum_p: sum of all beta values (see 'param_sum' function)
  
  Returns
  -------
  lr: logistic regression value
  """
  exp = np.exp(-sum_p)
  lr = 1/(1+exp)

  return lr

def log_reg_two(spectra, beta):
  """
  Description
  -----------
  Calculate logistic regression values for an array of spectra.

  Parameters
  ----------
  spectra: array of flux values from multiple spectra
  beta: beta values

  Returns
  -------
  lr: array of logistic regression function values
  """
  lr = []
  for i in spectra:
    z = param_sum(i, beta)
    exp = np.exp(-z)
    recip = 1/(1+exp)
    lr.append(recip)

  lr = np.asarray(lr)

  return lr

def gradient1(Y,sum_p):
  """
  Description
  -----------
  Used to calculate the gradient of the first beta value (no dependence on x).

  Parameters
  ----------
  Y: label  (0 or 1)
  sum_p: sum of all beta values (see 'param_sum' function)

  Returns
  -------
  num/denom: gradient value
  """
  if Y == 1:
    num = -np.exp(-sum_p)
    denom = 1 + np.exp(-sum_p)

  elif Y == 0:
    num = 1
    denom = 1 + np.exp(-sum_p)
  
  return num/denom

def gradient2(Y,x,sum_p):
  """
  Description
  -----------
  Used to  calculate the gradients of the beta values (excluding the first).

  Parameters
  ----------
  Y: label (0 or 1)
  x: flux value
  sum_p: sum of all beta values (see 'param_sum' function)

  Returns
  -------
  num/denom: gradient value
  """
  if Y == 1:
    num = -x * np.exp(-sum_p)
    denom = 1 + np.exp(-sum_p)

  elif Y == 0:
    num = x
    denom = 1 + np.exp(-sum_p)
  
  return num/denom

def loss(Y, spectra, beta, Yval, val_spec):
  """
  Description
  -----------
  Calculate the loss for a specfic set of beta values

  Parameters
  ----------
  Y: labels (0 or 1)
  spectra: flux values
  beta: beta values
  Yval: validation set labels (0 or 1)
  val_spec: validation flux values
  
  Returns
  -------
  J_sum: total loss calculated from all spectra
  beta_gradients: gradients for each beat value
  J_sum_val: validation loss calaculated from all validation spectra
  """
  J_total = []

  i = 0
  while i < len(Y):
    #do logistic regression
    sum_p = param_sum(spectra[i],beta)
    lr = log_reg(sum_p)

    #deal with log(0) cases
    if (lr == 1 and Y[i] == 0) or (lr == 0 and Y[i] == 1):
      J_iter = 1e+30

    else:  
      J_iter = (-Y[i]*np.log(lr)) - ((1-Y[i])*np.log(1-lr))

    J_total.append(J_iter)
    i += 1

  J_sum = (1/len(Y))*np.sum(J_total)
  
  J_total_val = []

  #validation
  i = 0
  while i < len(Yval):
    sum_p_val = param_sum(val_spec[i],beta)
    lr_val = log_reg(sum_p_val)

    if (lr_val == 1 and Yval[i] == 0) or (lr_val == 0 and Yval[i] == 1):
      J_iter_val = 1e+30

    else:  
      J_iter_val = (-Yval[i]*np.log(lr_val)) - ((1-Yval[i])*np.log(1-lr_val))

    J_total_val.append(J_iter_val)
    i += 1

  J_sum_val = (1/len(Yval))*np.sum(J_total_val)

  #shuffle the data for SGD
  Y, spectra = unison_shuffled_copies(Y, spectra)

  #select subset of data
  batch = 100
  Y_batch = Y[0:batch]
  spectra_batch = spectra[0:batch]
  beta_gradients = np.zeros(len(beta))
  i = 0

  #calculate gradients
  while i < len(Y_batch):
    sum_p = param_sum(spectra_batch[i],beta)
    for j in range(len(beta)):
      if j == 0:
        beta_gradients[j] += gradient1(Y_batch[i], sum_p)
      
      else:
        beta_gradients[j] += gradient2(Y_batch[i], spectra_batch[i][j-1], sum_p)

    i += 1

  return J_sum, beta_gradients, J_sum_val

def min_max_index(flux, outerVal=0):
  """
  Originally from https://github.com/daniel-muthukrishna/astrodash
  
  :param flux: 
  :param outerVal: is the scalar value in all entries before the minimum and after the maximum index
  :return: 
  """
  nonZeros = np.where(flux != outerVal)[0]
  if nonZeros.size:
      minIndex, maxIndex = min(nonZeros), max(nonZeros)
  else:
      minIndex, maxIndex = len(flux), len(flux)

  return minIndex, maxIndex