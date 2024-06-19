from astropy import units as u
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width

def eqw(a,b,sfWave,sfFlux):
    spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(sfFlux+1) * u.Unit('erg cm-2 s-1 AA-1'))
    peqw = equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value
    
    return peqw