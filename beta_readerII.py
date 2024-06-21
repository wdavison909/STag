import numpy as np

def beta_reader(direc):
    """
    Function to read in beta values for each tag
    """
    path = direc
    si_4000_beta = np.genfromtxt('%s/Beta Values II/si_4000_beta3.txt' % path)
    s_beta = np.genfromtxt('%s/Beta Values II/s_beta3.txt' % path)
    si_6150_beta = np.genfromtxt('%s/Beta Values II/si_6150_beta3.txt' % path)
    he_beta = np.genfromtxt('%s/Beta Values II/he_5876_beta.txt' % path)
    ha_beta = np.genfromtxt('%s/Beta Values II/ha_beta8.txt' % path)
    hb_beta = np.genfromtxt('%s/Beta Values II/hb_beta3.txt' % path)
    ca_beta = np.genfromtxt('%s/Beta Values II/ca_hk_beta3.txt' % path)
    fe_5170_beta = np.genfromtxt('%s/Beta Values II/fe_5170_beta3.txt' % path)
    haw_beta = np.genfromtxt('%s/Beta Values II/ha_wide_beta.txt' % path)

    return si_4000_beta,s_beta,si_6150_beta,he_beta,ha_beta,hb_beta,ca_beta,fe_5170_beta,haw_beta