#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:03:35 2021

@author: willdavison
"""

"""
What follows has been taken from the DASH software GitHub, with relevant 
modifications indicated. For a more complete understanding please visit
(https://github.com/daniel-muthukrishna/astrodash).
"""

import os
import sys
import numpy as np
import astropy.io.fits as afits
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import medfilt
from scipy.integrate import cumtrapz

try:
    import pandas as pd

    USE_PANDAS = True
except ImportError:
    print("Pandas module not installed. DASH will use numpy to load spectral files instead. "
          "This can be up to 10x slower.")
    USE_PANDAS = False


class ProcessingTools(object):
    def redshift_spectrum(self, wave, flux, z):
        wave_new = wave * (z + 1)

        return wave_new, flux

    def deredshift_spectrum(self, wave, flux, z):
        wave_new = wave / (z + 1)

        return wave_new, flux

    def min_max_index(self, flux, outerVal=0):
        """ 
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


class ReadSpectrumFile(object):
    def __init__(self, filename, w0, w1, nw):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.processingTools = ProcessingTools()

    def read_dat_file(self):
        try:
            if USE_PANDAS is True:
                data = pd.read_csv(self.filename, header=None, delim_whitespace=True).values
            else:
                data = np.loadtxt(self.filename)
            wave = data[:, 0]
            flux = data[:, 1]
        except:
            print("COULDN'T USE LOADTXT FOR FILE: {0}\n READ LINE BY LINE INSTEAD.".format(self.filename))
            wave = []
            flux = []
            with open(self.filename, 'r') as FileObj:
                for line in FileObj:
                    if line.strip() != '' and line.strip()[0] != '#':
                        datapoint = line.rstrip('\n').strip().split()
                        wave.append(float(datapoint[0].replace('D', 'E')))
                        flux.append(float(datapoint[1].replace('D', 'E')))

            wave = np.array(wave)
            flux = np.array(flux)

        sorted_indexes = np.argsort(wave)
        wave = wave[sorted_indexes]
        flux = flux[sorted_indexes]

        return wave, flux

    def file_extension(self, template=False):
        if isinstance(self.filename, (list, np.ndarray)):  # Is an Nx2 array
            wave, flux = self.filename[:,0], self.filename[:,1]
            return wave, flux
        elif hasattr(self.filename, 'read'):  # Is a file handle
            self.filename.seek(0)
            return self.read_dat_file()
        else:  # Is a filename string
            filename = os.path.basename(self.filename)
            extension = filename.split('.')[-1]

            if extension == self.filename or extension in ['flm', 'txt', 'dat']:
                return self.read_dat_file()
            else:
                try:
                    return self.read_dat_file()
                except:
                    print("Invalid Input File")
                    return 0

    def two_col_input_spectrum(self, wave, flux, z):
        wave, flux = self.processingTools.deredshift_spectrum(wave, flux, z)

        mask = (wave >= self.w0) & (wave < self.w1)
        wave = wave[mask]
        flux = flux[mask]

        if not wave.any():
            raise Exception("The spectrum {0} with redshift {1} is out of the wavelength range {2}A to {3}A, "
                            "and cannot be classified. Please remove this object or change the input redshift of this"
                            " spectrum.".format(self.filename, z, int(self.w0), int(self.w1)))

        fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

        return wave, fluxNorm

def zero_non_overlap_part(array, minIndex, maxIndex, outerVal=0.):
    slicedArray = np.copy(array)
    slicedArray[0:minIndex] = outerVal * np.ones(minIndex)
    slicedArray[maxIndex:] = outerVal * np.ones(len(array) - maxIndex)

    return slicedArray


def normalise_spectrum(flux):
    if len(flux) == 0 or min(flux) == max(flux):  # No data
        fluxNorm = np.zeros(len(flux))
    else:
        fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

    return fluxNorm
    
class PreProcessSpectrum(object):
    def __init__(self, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.dwlog = np.log(w1 / w0) / nw
        self.processingTools = ProcessingTools()

    def log_wavelength(self, wave, flux):
        # Set up log wavelength array bins
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * self.dwlog)

        fluxOut = self._vectorised_log_binning(wave, flux)

        minIndex, maxIndex = self.processingTools.min_max_index(fluxOut, outerVal=0)

        return wlog, fluxOut, minIndex, maxIndex

    def _vectorised_log_binning(self, wave, flux):
        """ Vectorised code version of the self._original_log_binning (for improved speed since this is the most called
        function in the script during training). This is complicated, but it has been tested to match the slower
        looping method """

        spec = np.array([wave, flux]).T
        mask = (wave >= self.w0) & (wave < self.w1)
        spec = spec[mask]
        wave, flux = spec.T
        try:
            fluxOut = np.zeros(int(self.nw))
            waveMiddle = wave[1:-1]
            waveTake1Index = wave[:-2]
            wavePlus1Index = wave[2:]
            s0List = 0.5 * (waveTake1Index + waveMiddle)
            s1List = 0.5 * (waveMiddle + wavePlus1Index)
            s0First = 0.5 * (3 * wave[0] - wave[1])
            s0Last = 0.5 * (wave[-2] + wave[-1])
            s1First = 0.5 * (wave[0] + wave[1])
            s1Last = 0.5 * (3 * wave[-1] - wave[-2])
            s0List = np.concatenate([[s0First], s0List, [s0Last]])
            s1List = np.concatenate([[s1First], s1List, [s1Last]])
            s0LogList = np.log(s0List / self.w0) / self.dwlog + 1
            s1LogList = np.log(s1List / self.w0) / self.dwlog + 1
            dnuList = s1List - s0List

            s0LogListInt = s0LogList.astype(int)
            s1LogListInt = s1LogList.astype(int)
            numOfJLoops = s1LogListInt - s0LogListInt
            jIndexes = np.flatnonzero(numOfJLoops)
            jIndexVals = s0LogListInt[jIndexes]
            prependZero = jIndexVals[0] if jIndexVals[0] < 0 else False
            if prependZero is not False:
                jIndexVals[0] = 0
                numOfJLoops[0] += prependZero
            numOfJLoops = (numOfJLoops[jIndexes])[jIndexVals < self.nw]
            fluxValList = ((flux * 1 / (s1LogList - s0LogList) * dnuList)[jIndexes])[jIndexVals < self.nw]
            fluxValList = np.repeat(fluxValList, numOfJLoops)
            minJ = min(jIndexVals)
            maxJ = (max(jIndexVals) + numOfJLoops[-1]) if (max(jIndexVals) + numOfJLoops[-1] < self.nw) else self.nw
            fluxOut[minJ:maxJ] = fluxValList[:(maxJ - minJ)]

            return fluxOut
        except Exception as e:
            print(e)
            print('wave', wave)
            print('flux', flux)
            print("########################################ERROR#######################################\n\n\n\n")
            return np.zeros(self.nw)

    def spline_fit(self, wave, flux, numSplinePoints, minindex, maxindex):
        continuum = np.zeros(int(self.nw)) + 1
        if (maxindex - minindex) > 5:
            spline = UnivariateSpline(wave[minindex:maxindex + 1], flux[minindex:maxindex + 1], k=3)
            splineWave = np.linspace(wave[minindex], wave[maxindex], num=numSplinePoints, endpoint=True)
            splinePoints = spline(splineWave)

            splineMore = UnivariateSpline(splineWave, splinePoints, k=3)
            splinePointsMore = splineMore(wave[minindex:maxindex + 1])

            continuum[minindex:maxindex + 1] = splinePointsMore
        else:
            print("WARNING: LESS THAN 6 POINTS IN SPECTRUM")

        return continuum

    def continuum_removal(self, wave, flux, numSplinePoints, minIndex, maxIndex):
        flux = flux + 1  # Important to keep this as +1
        contRemovedFlux = np.copy(flux)

        splineFit = self.spline_fit(wave, flux, numSplinePoints, minIndex, maxIndex)
        contRemovedFlux[minIndex:maxIndex + 1] = flux[minIndex:maxIndex + 1] / splineFit[minIndex:maxIndex + 1]
        contRemovedFluxNorm = normalise_spectrum(contRemovedFlux - 1)
        contRemovedFluxNorm = zero_non_overlap_part(contRemovedFluxNorm, minIndex, maxIndex)

        return contRemovedFluxNorm, splineFit - 1

    def mean_zero(self, flux, minindex, maxindex):
        """mean zero flux"""
        meanflux = np.mean(flux[minindex:maxindex])
        varflux = np.std(flux[minindex:maxindex]) #this line has been added for scaling
        meanzeroflux = (flux - meanflux)/varflux
        meanzeroflux[0:minindex] = flux[0:minindex]
        meanzeroflux[maxindex + 1:] = flux[maxindex + 1:]

        return meanzeroflux

    def apodize(self, flux, minindex, maxindex, outerVal=0):
        """apodize with 5% cosine bell"""
        percent = 0.05
        fluxout = np.copy(flux) - outerVal

        nsquash = int(self.nw * percent)
        for i in range(0, nsquash):
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1 - np.cos(arg))
            if (minindex + i < self.nw) and (maxindex - i >= 0):
                fluxout[minindex + i] = factor * fluxout[minindex + i]
                fluxout[maxindex - i] = factor * fluxout[maxindex - i]
            else:
                print("INVALID FLUX IN PREPROCESSING.PY APODIZE()")
                print("MININDEX=%d, i=%d" % (minindex, i))
                break

        if outerVal != 0:
            fluxout = fluxout + outerVal
            fluxout = zero_non_overlap_part(fluxout, minindex, maxindex, outerVal=outerVal)

        return fluxout

def limit_wavelength_range(wave, flux, minWave, maxWave):
    minIdx = (np.abs(wave - minWave)).argmin()
    maxIdx = (np.abs(wave - maxWave)).argmin()

    flux[:minIdx] = np.zeros(minIdx)
    flux[maxIdx:] = np.zeros(len(flux) - maxIdx)

    return flux

class PreProcessing(object):
    """ Pre-processes spectra before training """

    def __init__(self, filename, w0, w1, nw):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.numSplinePoints = 13
        self.processingTools = ProcessingTools()
        self.readSpectrumFile = ReadSpectrumFile(filename, w0, w1, nw)
        self.preProcess = PreProcessSpectrum(w0, w1, nw)

        self.spectrum = self.readSpectrumFile.file_extension()
        if len(self.spectrum) == 3:
            self.redshiftFromFile = True
        else:
            self.redshiftFromFile = False

    def two_column_data(self, z, smooth, minWave, maxWave):
        if self.redshiftFromFile is True:
            self.wave, self.flux, z = self.spectrum
        else:
            self.wave, self.flux = self.spectrum
        self.flux = normalise_spectrum(self.flux)
        self.flux = limit_wavelength_range(self.wave, self.flux, minWave, maxWave)
        self.wDensity = (self.w1 - self.w0) / self.nw  # Average wavelength spacing
        wavelengthDensity = (max(self.wave) - min(self.wave)) / len(self.wave)
        filterSize = int(self.wDensity / wavelengthDensity * smooth/ 2) * 2 + 1
        preFiltered = medfilt(self.flux, kernel_size=filterSize)
        wave, deredshifted = self.readSpectrumFile.two_col_input_spectrum(self.wave, preFiltered, z)
        if len(wave) < 2:
            sys.exit("The redshifted spectrum of file: {0} is out of the classification range between {1} to {2} "
                     "Angstroms. Please remove this file from classification or reduce the redshift before re-running "
                     "the program.".format(self.filename, self.w0, self.w1))

        binnedwave, binnedflux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, deredshifted)
        newflux, continuum = self.preProcess.continuum_removal(binnedwave, binnedflux, self.numSplinePoints, minIndex, maxIndex)
        meanzero = self.preProcess.mean_zero(newflux, minIndex, maxIndex)
        apodized = self.preProcess.apodize(meanzero, minIndex, maxIndex)
        area = cumtrapz(np.abs(apodized),x=binnedwave) #Find area under the spectrum
        norm_area = area[-1]/(binnedwave[-1] - binnedwave[0]) #Normalise the area

        return binnedwave, apodized, minIndex, maxIndex, z, norm_area