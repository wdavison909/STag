import numpy as np
from astropy.io import fits
import spectra_preprocessing as sp
from astropy import units as u
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from tensorflow import keras
from scipy.fftpack import fft
from scipy.signal import argrelmax

def log_reg_two(spectra, beta):
    lr = []
    for i in spectra:
        z = param_sum(i, beta)
        exp = np.exp(-z)
        recip = 1/(1+exp)
        lr.append(recip)

    lr = np.asarray(lr)

    return lr

def param_sum(x,beta):
    sum_p = beta[0]
    for i in range(len(x)):
        sum_p += beta[i+1]*x[i]

    return sum_p

def cross_correlation(inputFlux, templateFlux):
        inputfourier = fft(inputFlux)
        tempfourier = fft(templateFlux)

        product = inputfourier * np.conj(tempfourier)
        xCorr = fft(product)

        rmsInput = np.std(inputfourier)
        rmsTemp = np.std(tempfourier)

        xCorrNorm = (1. / (1500 * rmsInput * rmsTemp)) * xCorr

        rmsXCorr = np.std(product)

        xCorrNormRearranged = np.concatenate(
            (xCorrNorm[int(len(xCorrNorm) / 2):], xCorrNorm[0:int(len(xCorrNorm) / 2)]))

        crossCorr = np.correlate(inputFlux, templateFlux, mode='Full')[::-1][
                    int(1500 / 2):int(1500 + 1500 / 2)] / max(
            np.correlate(inputFlux, templateFlux, mode='Full'))

        try:
            deltapeak, h = get_peaks(crossCorr)[0]
            shift = int(deltapeak - 1500 / 2)
            autoCorr = np.correlate(templateFlux, templateFlux, mode='Full')[::-1][
                       int(1500 / 2) - shift:int(1500 + 1500 / 2) - shift] / max(
                np.correlate(templateFlux, templateFlux, mode='Full'))

            aRandomFunction = crossCorr - autoCorr
            rmsA = np.std(aRandomFunction)
        except IndexError as err:
            print("Error: Cross-correlation is zero, probably caused by empty spectrum.", err)
            rmsA = 1

        return xCorr, rmsInput, rmsTemp, xCorrNorm, rmsXCorr, xCorrNormRearranged, rmsA

def get_peaks(crosscorr):
        peakindexes = argrelmax(crosscorr)[0]

        ypeaks = []
        for i in peakindexes:
            ypeaks.append(abs(crosscorr[i]))

        arr = list(zip(*[peakindexes, ypeaks]))
        arr.sort(key=lambda x: x[1])
        sortedPeaks = list(reversed(arr))

        return sortedPeaks

def calculate_r(crosscorr, rmsA):
        deltapeak1, h1 = get_peaks(crosscorr)[0]  # deltapeak = np.argmax(abs(crosscorr))
        deltapeak2, h2 = get_peaks(crosscorr)[1]

        r = abs((h1 - rmsA) / (np.sqrt(2) * rmsA))
        fom = (h1 - 0.05) ** 0.75 * (h1 / h2)

        return r, deltapeak1, fom 

def calculate_rlap(crosscorr, rmsAntisymmetric, inputFlux, templateFlux, wave):
        r, deltapeak, fom = calculate_r(crosscorr, rmsAntisymmetric)
        shift = int(deltapeak - 1500 / 2)  # shift from redshift

        # lap value
        iminindex, imaxindex = min_max_index(inputFlux)
        tminindex, tmaxindex = min_max_index(templateFlux)

        overlapminindex = int(max(iminindex + shift, tminindex))
        overlapmaxindex = int(min(imaxindex - 1 + shift, tmaxindex - 1))

        minWaveOverlap = wave[overlapminindex]
        maxWaveOverlap = wave[overlapmaxindex]

        lap = np.log(maxWaveOverlap / minWaveOverlap)
        rlap = 5 * r * lap

        fom = fom * lap
        # print r, lap, rlap, fom
        return r, lap, rlap, fom

def min_max_index(flux):
        minindex, maxindex = (0, 1500 - 1)
        zeros = np.where(flux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                break
            j += 1
            minindex = j
        j = int(1500) - 1
        for i in zeros[::-1]:
            if (i != j):
                break
            j -= 1
            maxindex = j

        return minindex, maxindex
    
def z_check(path,path2,ID,z_all,raw_spec,raw_wave,ppspec,wave,preds,si_4000_beta,s_beta,si_6150_beta,he_beta,ha_beta,hb_beta,ca_beta,fe_5170_beta,haw_beta,ha_result,hb_result,ca_result,si_4000_result,si_6150_result,s_result,he_result,fe_result,haw_result,ha_eqw,hb_eqw,ca_eqw,si4_eqw,si6_eqw,he_eqw,fe_eqw,haw_eqw):
    #Load Ia models
    ia_neg = np.load('%s/models_ia_-10.npy' % path2)
    ia_0 = np.load('%s/models_ia_0.npy' % path2)
    ia_7 = np.load('%s/models_ia_7.npy' % path2)
    ia_14 = np.load('%s/models_ia_14.npy' % path2)
    ia_21 = np.load('%s/models_ia_21.npy' % path2)
    #Load Ib models
    ib_neg = np.load('%s/models_ib_-10.npy' % path2)
    ib_0 = np.load('%s/models_ib_0.npy' % path2)
    ib_7 = np.load('%s/models_ib_7.npy' % path2)
    ib_14 = np.load('%s/models_ib_14.npy' % path2)
    ib_21 = np.load('%s/models_ib_21.npy' % path2)
    #Load Ic models
    ic_neg = np.load('%s/models_ic_-10.npy' % path2)
    ic_0 = np.load('%s/models_ic_0.npy' % path2)
    ic_7 = np.load('%s/models_ic_7.npy' % path2)
    ic_14 = np.load('%s/models_ic_14.npy' % path2)
    ic_21 = np.load('%s/models_ic_21.npy' % path2)
    #Load II models
    ii_neg = np.load('%s/models_ii_-10.npy' % path2)
    ii_0 = np.load('%s/models_ii_0.npy' % path2)
    ii_7 = np.load('%s/models_ii_7.npy' % path2)
    ii_14 = np.load('%s/models_ii_14.npy' % path2)
    ii_21 = np.load('%s/models_ii_21.npy' % path2)

    cuts = np.genfromtxt('%s/cuts_v2.txt' % path, dtype=int)
    si_6150_cuts = cuts[0]
    ca_cuts = cuts[1]
    s_cuts = cuts[2]
    ha_cuts = cuts[3]
    he_5876_cuts = cuts[4]
    fe_5170_cuts = cuts[5]
    si_4000_cuts = cuts[6]
    hb_cuts = cuts[7]
    haw_cuts = cuts[8]
    
    model = keras.models.load_model('%s/STagV4_model_v1' % path)
    
    all_rlaps_mean = []
    phase_bin = []
    for i in range(len(preds)):
        rlaps_mean = []
        iF = ppspec[i]

        if preds[i] == 0:
            rlaps = []
            for j in range(len(ia_neg)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_neg[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_neg[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ia_0)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_0[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_0[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ia_7)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_7[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_7[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps)) 

            rlaps = []
            for j in range(len(ia_14)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_14[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_14[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ia_21)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_21[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_21[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

        elif preds[i] == 1:
            rlaps = []
            for j in range(len(ib_neg)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_neg[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_neg[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ib_0)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_0[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_0[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ib_7)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_7[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_7[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ib_14)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_14[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_14[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ib_21)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_21[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_21[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))    

        elif preds[i] == 2:
            rlaps = []
            for j in range(len(ic_neg)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_neg[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_neg[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ic_0)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_0[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_0[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ic_7)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_7[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_7[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps)) 

            rlaps = []
            for j in range(len(ic_14)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_14[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_14[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ic_21)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_21[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_21[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))    

        elif preds[i] == 3:
            rlaps = []
            for j in range(len(ii_neg)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_neg[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_neg[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ii_0)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_0[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_0[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ii_7)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_7[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_7[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps)) 

            rlaps = []
            for j in range(len(ii_14)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_14[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_14[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

            rlaps = []
            for j in range(len(ii_21)):
                xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_21[j])
                r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_21[j],wave)
                rlaps.append(rlap)

            rlaps_mean.append(np.mean(rlaps))

        all_rlaps_mean.append(np.max(rlaps_mean))
        phase_bin.append(np.argmax(rlaps_mean))

    phase_bin_real = []
    for l in range(len(phase_bin)):
        if phase_bin[l] == 0:
            phase_bin_real.append('-10 < t < 0')
        elif phase_bin[l] == 1:
            phase_bin_real.append('0 <= t <= 7')
        elif phase_bin[l] == 2:
            phase_bin_real.append('7 < t <= 14')
        elif phase_bin[l] == 3:
            phase_bin_real.append('14 < t <= 21')
        elif phase_bin[l] == 4:
            phase_bin_real.append('21 < t')    

    final_id = []
    final_z = []
    final_class = []
    final_phase = []
    final_rlap = []
    final_spec = []
    final_tags = []
    final_eqw = []
    z_change = []
    scan_id = []
    scan_z = []
    scan_wave = []
    scan_spec = []
    
    for j in range(len(all_rlaps_mean)):
        if all_rlaps_mean[j] >= 6:
            final_id.append(ID[j])
            final_z.append(z_all[j])
            final_class.append(preds[j])
            final_phase.append(phase_bin_real[j])
            final_rlap.append(all_rlaps_mean[j])
            final_spec.append(ppspec[j])
            final_tags.append([ha_result[j],hb_result[j],ca_result[j],si_4000_result[j],si_6150_result[j],s_result[j],he_result[j],fe_result[j],haw_result[j]])
            final_eqw.append([ha_eqw[j],hb_eqw[j],ca_eqw[j],si4_eqw[j],si6_eqw[j],he_eqw[j],fe_eqw[j],haw_eqw[j]])
            z_change.append('No')
        elif all_rlaps_mean[j] < 6:
            scan_id.append(ID[j])
            scan_z.append(z_all[j])
            scan_wave.append(raw_wave[j])
            scan_spec.append(raw_spec[j])

    for m in range(len(scan_id)):
        scan_ppspec = []
        zrange = (1 + scan_z[m]) * 0.1
        minz = scan_z[m] - zrange/2
        maxz = scan_z[m] + zrange/2

        ztest = []
        zprime = np.linspace(minz,maxz,11)
        for i in range(len(zprime)):
            if zprime[i] >= 0:
                ztest.append(zprime[i])

        index2 = 5 - (11 - len(ztest))

        full = np.column_stack((scan_wave[m], scan_spec[m]))

        #Initialise for pre-processing
        preProcess = sp.PreProcessing(full, 2500, 10000, 1500)

        for n in range(len(ztest)):
            #Do the pre-processing steps
            sfWave, sfFlux, minInd, maxInd, sfZ, sfArea = preProcess.two_column_data(ztest[n], smooth=6, minWave=2500, maxWave=10000)

            scan_ppspec.append(sfFlux)

        scan_si6_eqw = []
        scan_ca_eqw = []
        scan_ha_eqw = []
        scan_he_eqw = []
        scan_fe_eqw = []
        scan_si4_eqw = []
        scan_hb_eqw = []
        scan_haw_eqw = []
        for o in range(len(scan_ppspec)):
            #Si 6150 pEW
            a = si_6150_cuts[0]
            b = si_6150_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_si6_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #Ca H&K pEW
            a = ca_cuts[0]
            b = ca_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_ca_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #H alpha pEW
            a = ha_cuts[0]
            b = ha_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_ha_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #He 5876 pEW
            a = he_5876_cuts[0]
            b = he_5876_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_he_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #Fe 5170 pEW
            a = fe_5170_cuts[0]
            b = fe_5170_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_fe_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #Si 4000 pEW
            a = si_4000_cuts[0]
            b = si_4000_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_si4_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #H beta pEW
            a = hb_cuts[0]
            b = hb_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_hb_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

            #H alpha wide pEW
            a = haw_cuts[0]
            b = haw_cuts[1]
            spec = Spectrum1D(spectral_axis=sfWave * u.AA, flux=(scan_ppspec[o]+1) * u.Unit('erg cm-2 s-1 AA-1'))
            scan_haw_eqw.append(equivalent_width(spec, continuum=1, regions=SpectralRegion(sfWave[a] * u.AA, sfWave[b] * u.AA)).value)

        scan_ppspec = np.asarray(scan_ppspec)

        #Cut spectra
        si_6150_cut_spectra = scan_ppspec[:,si_6150_cuts[0]:si_6150_cuts[1]]
        ca_cut_spectra = scan_ppspec[:,ca_cuts[0]:ca_cuts[1]]
        s_cut_spectra = scan_ppspec[:,s_cuts[0]:s_cuts[1]]
        ha_cut_spectra = scan_ppspec[:,ha_cuts[0]:ha_cuts[1]]
        he_cut_spectra = scan_ppspec[:,he_5876_cuts[0]:he_5876_cuts[1]]
        fe_cut_spectra = scan_ppspec[:,fe_5170_cuts[0]:fe_5170_cuts[1]]
        si_4000_cut_spectra = scan_ppspec[:,si_4000_cuts[0]:si_4000_cuts[1]]
        hb_cut_spectra = scan_ppspec[:,hb_cuts[0]:hb_cuts[1]]
        haw_cut_spectra = scan_ppspec[:,haw_cuts[0]:haw_cuts[1]]

        scan_network_test = np.zeros([len(scan_ppspec),18])

        #Assign tag probablities
        #H alpha
        scan_ha_result = log_reg_two(ha_cut_spectra, ha_beta)
        scan_network_test[:,0] = scan_ha_result

        #H beta
        scan_hb_result = log_reg_two(hb_cut_spectra, hb_beta)
        scan_network_test[:,1] = scan_hb_result

        #Si 4000
        scan_si_4000_result = log_reg_two(si_4000_cut_spectra, si_4000_beta)
        scan_network_test[:,2] = scan_si_4000_result

        #Si 6150
        scan_si_6150_result = log_reg_two(si_6150_cut_spectra, si_6150_beta)
        scan_network_test[:,3] = scan_si_6150_result

        #S W
        scan_s_result = log_reg_two(s_cut_spectra, s_beta)
        scan_network_test[:,4] = scan_s_result

        #He 5876
        scan_he_result = log_reg_two(he_cut_spectra, he_beta)
        scan_network_test[:,5] = scan_he_result

        #Ca H&K
        scan_ca_result = log_reg_two(ca_cut_spectra, ca_beta)
        scan_network_test[:,6] = scan_ca_result

        #Fe 5170
        scan_fe_result = log_reg_two(fe_cut_spectra, fe_5170_beta)
        scan_network_test[:,7] = scan_fe_result

        #He Eq. Width
        scan_network_test[:,8] = scan_he_eqw

        #H alpha Eq. Width
        scan_network_test[:,9] = scan_ha_eqw

        #H beta Eq. Width
        scan_network_test[:,10] = scan_hb_eqw

        #Ca HK Eq. Width
        scan_network_test[:,11] = scan_ca_eqw

        #Si 4000 Eq. Width
        scan_network_test[:,12] = scan_si4_eqw

        #Si 4000 prob. * eqw
        scan_network_test[:,13]  = scan_si_4000_result * scan_si4_eqw

        #Fe 5170 Eq. Width
        scan_network_test[:,14] = scan_fe_eqw

        #Si 6150 Eq. Width
        scan_network_test[:,15] = scan_si6_eqw

        #H alpha wide
        scan_haw_result = log_reg_two(haw_cut_spectra, haw_beta)
        scan_network_test[:,16] = scan_haw_result

        #H alpha wide Eq. Width
        scan_network_test[:,17] = scan_haw_eqw

        #Predict classes
        scan_class_prob = model.predict(scan_network_test)
        scan_preds = np.argmax(scan_class_prob, axis=-1)

        scan_all_rlaps_mean = []
        scan_phase_bin = []
        for i in range(len(scan_preds)):
            rlaps_mean = []
            iF = scan_ppspec[i]

            if scan_preds[i] == 0:
                rlaps = []
                for j in range(len(ia_neg)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_neg[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_neg[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ia_0)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_0[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_0[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ia_7)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_7[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_7[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps)) 

                rlaps = []
                for j in range(len(ia_14)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_14[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_14[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ia_21)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ia_21[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ia_21[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

            elif scan_preds[i] == 1:
                rlaps = []
                for j in range(len(ib_neg)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_neg[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_neg[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ib_0)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_0[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_0[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ib_7)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_7[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_7[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ib_14)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_14[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_14[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ib_21)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ib_21[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ib_21[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))    

            elif scan_preds[i] == 2:
                rlaps = []
                for j in range(len(ic_neg)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_neg[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_neg[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ic_0)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_0[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_0[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ic_7)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_7[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_7[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps)) 

                rlaps = []
                for j in range(len(ic_14)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_14[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_14[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ic_21)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ic_21[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ic_21[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))    

            elif scan_preds[i] == 3:
                rlaps = []
                for j in range(len(ii_neg)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_neg[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_neg[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ii_0)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_0[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_0[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ii_7)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_7[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_7[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps)) 

                rlaps = []
                for j in range(len(ii_14)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_14[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_14[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

                rlaps = []
                for j in range(len(ii_21)):
                    xc, rmsI, rmsT, xcNorm, xcRMS, xcNormRe, rmsA = cross_correlation(iF,ii_21[j])
                    r, lap, rlap, fom = calculate_rlap(xcNormRe,rmsA,iF,ii_21[j],wave)
                    rlaps.append(rlap)

                rlaps_mean.append(np.mean(rlaps))

            scan_all_rlaps_mean.append(np.max(rlaps_mean))
            scan_phase_bin.append(np.argmax(rlaps_mean))

        scan_phase_bin_real = []
        for l in range(len(scan_phase_bin)):
            if scan_phase_bin[l] == 0:
                scan_phase_bin_real.append('-10 < t < 0')
            elif scan_phase_bin[l] == 1:
                scan_phase_bin_real.append('0 <= t <= 7')
            elif scan_phase_bin[l] == 2:
                scan_phase_bin_real.append('7 < t <= 14')
            elif scan_phase_bin[l] == 3:
                scan_phase_bin_real.append('14 < t <= 21')
            elif scan_phase_bin[l] == 4:
                scan_phase_bin_real.append('21 < t')

        comp = 0
        index = 'None'

        for p in range(len(scan_all_rlaps_mean)):
            if scan_all_rlaps_mean[p] >= 6 and scan_all_rlaps_mean[p] > comp:
                comp = scan_all_rlaps_mean[p]
                index = p

        if index != 'None':
            final_id.append(scan_id[m])
            final_z.append(ztest[index])
            final_class.append(scan_preds[index])
            final_phase.append(scan_phase_bin_real[index])
            final_rlap.append(scan_all_rlaps_mean[index])
            final_spec.append(scan_ppspec[index])
            final_tags.append([scan_ha_result[index],scan_hb_result[index],scan_ca_result[index],scan_si_4000_result[index],scan_si_6150_result[index],scan_s_result[index],scan_he_result[index],scan_fe_result[index],scan_haw_result[index]])
            final_eqw.append([scan_ha_eqw[index],scan_hb_eqw[index],scan_ca_eqw[index],scan_si4_eqw[index],scan_si6_eqw[index],scan_he_eqw[index],scan_fe_eqw[index],scan_haw_eqw[index]])
            z_change.append('Yes')
        elif index == 'None':
            final_id.append(scan_id[m])
            final_z.append(scan_z[m])
            final_class.append('-')
            final_phase.append('-')
            final_rlap.append('-')
            final_spec.append(scan_ppspec[index2])        
            final_tags.append([scan_ha_result[index2],scan_hb_result[index2],scan_ca_result[index2],scan_si_4000_result[index2],scan_si_6150_result[index2],scan_s_result[index2],scan_he_result[index2],scan_fe_result[index2],scan_haw_result[index2]])
            final_eqw.append([scan_ha_eqw[index2],scan_hb_eqw[index2],scan_ca_eqw[index2],scan_si4_eqw[index2],scan_si6_eqw[index2],scan_he_eqw[index2],scan_fe_eqw[index2],scan_haw_eqw[index2]])
            z_change.append('-')

    for k in range(len(final_id)):
        if final_class[k] != '-':
            print('SN ID: %s | z: %.3f | Class: %d | rlap: %.3f | z change?: %s | Phase: %s' % (final_id[k],final_z[k],final_class[k],final_rlap[k],z_change[k],final_phase[k]))
            print('n-HA: %.2f | HA: %.2f | HB: %.2f | Ca H&K: %.2f | Si 4000: %.2f | Si 6355: %.2f | S: %.2f | He: %.2f | Fe: %.2f' % (final_tags[k][0],final_tags[k][8],final_tags[k][1],final_tags[k][2],final_tags[k][3],final_tags[k][4],final_tags[k][5],final_tags[k][6],final_tags[k][7]))
        else:
            print('SN ID: %s | z: %.3f | *** WARNING*** rlap < 6: NO CLASSIFICATION - CHECK REDSHIFT' % (final_id[k],final_z[k]))
            print('n-HA: %.2f | HA: %.2f | HB: %.2f | Ca H&K: %.2f | Si 4000: %.2f | Si 6355: %.2f | S: %.2f | He: %.2f | Fe: %.2f' % (final_tags[k][0],final_tags[k][8],final_tags[k][1],final_tags[k][2],final_tags[k][3],final_tags[k][4],final_tags[k][5],final_tags[k][6],final_tags[k][7]))