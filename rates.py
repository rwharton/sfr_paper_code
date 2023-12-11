import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

from matplotlib import rc
rc('font',**{'family':'DeJavu Serif',
              'serif':['Computer Modern Roman']})
rc('text', usetex=True)



def get_det_frac(Edet, Emin, Emax, gamma):
    """
    Get frac of detectable bursts assuming 
    power law
    """
    E0 = Emin
    if Edet > Emin:
        E0 = Edet
    else: pass

    if gamma == -1:
        fnum = np.log(E0 / Emax)
        fdem = np.log(Emin / Emax)
        fdet = fnum / fdem

    else:
        pow = gamma + 1
        fnum = (E0/Emax)**pow - 1
        fdem = (Emin/Emax)**pow - 1
        fdet = fnum / fdem

    return fdet


def get_det_frac_arr(Edet, Emin, Emax, gammas):
    ff = np.array([ get_det_frac(Edet, Emin, Emax, gamma) \
                    for gamma in gammas ])
    return ff


def get_det_frac_arr2(Edets, Emin, Emax, gamma):
    ff = np.array([ get_det_frac(Edet, Emin, Emax, gamma) \
                    for Edet in Edets ])
    return ff


def get_nc_avg_frac(Edet0, Emin, Emax, gamma):
    """
    Edet0 is A = 1 
    """
    Alist = np.array([0.55, 0.72, 0.88, 0.93, 0.88, 0.72, 0.55])
    flist = []
    for A in Alist:
        fii = get_det_frac(Edet0/A, Emin, Emax, gamma)
        flist.append(fii)

    flist = np.array(flist)
    favg = np.mean(flist)

    return favg


def calc_mag_hr_nc(Emin, Emax, gamma):
    """
    Calculate sum (Nmag * T * fdet) 
    """
    SFRs = np.array([ 0.35, 2.8,  13, 2.9, 4.3, 2.8, 2.7 ])
    Tobs = np.array([   51, 102, 184,  96, 115,  84,  63 ])
    Ds   = np.array([ 0.79, 3.3, 3.5, 6.4, 7.7, 7.8, 11.1])

    BW = 100.
    Emin0s = 5.7e34 * (BW/100) * (Ds/3.5)**2 

    fdets = []
    for Emin0 in Emin0s:
        fdii = get_nc_avg_frac(Emin0, Emin, Emax, gamma)
        fdets.append(fdii)

    fdets = np.array(fdets)

    mag_hr = 30 * (SFRs / 1.65) * fdets * Tobs

    return mag_hr
    

def calc_mag_lim_nc(Emin, Emax, gamma, BW=100):
    """
    Calculate sum (Nmag * T * fdet) 
    """
    SFRs = np.array([ 0.35, 2.8,  13, 2.9, 4.3, 2.8, 2.7 ])
    Tobs = np.array([   51, 102, 184,  96, 115,  84,  63 ])
    Ds   = np.array([ 0.79, 3.3, 3.5, 6.4, 7.7, 7.8, 11.1])

    Emin0s = 5.7e34 * (BW/100) * (Ds/3.5)**2 

    fdets = []
    for Emin0 in Emin0s:
        fdii = get_nc_avg_frac(Emin0, Emin, Emax, gamma)
        fdets.append(fdii)

    fdets = np.array(fdets)

    mag_hr = 30 * (SFRs / 1.65) * fdets * Tobs

    r_lim = (2.996 / np.sum(mag_hr)) * 365.25 * 24

    return r_lim
    

def calc_mag_lim_nc_m82(Emin, Emax, gamma, BW=100):
    """
    Calculate sum (Nmag * T * fdet) 
    """
    SFR  = 13
    Tobs = 184 
    D    = 3.6

    Emin0 = 5.7e34 * (BW/100) * (D/3.5)**2 
    fdet = get_nc_avg_frac(Emin0, Emin, Emax, gamma)
    mag_hr = 30 * (SFR / 1.65) * fdet * Tobs
    r_lim = (2.996 / mag_hr) * 365.25 * 24

    return r_lim
    
    


#################
###   PLOTS   ###
#################

def myLogFormat(y, pos):
    # Find the number of decimal places required
    # =0 for numbers >=1
    decimalplaces = int(np.maximum(-np.log10(y),0))
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


def make_f_gammas():
    Emin_nb = 7e34
    Emin_bb = 2.4e35
    Emax = 1e41

    SFR_M77 = 30
    SFR_M82 = 13 

    Nmag_M77 = 30 * (SFR_M77/1.65)
    Nmag_M82 = 30 * (SFR_M82/1.65)
    
    R95_M77 = 273 # per year
    R95_M82 = 233 # per year 

    # r_mag = R95 / (Nmag * f(E))

    gams = np.linspace(-5, 0, 1000)
    
    ff_nb = get_det_frac_arr(1.8, 1, Emax/Emin_nb, gams)
    ff_bb = get_det_frac_arr(11, 1, Emax/Emin_bb, gams)
    ff_1  = get_det_frac_arr(1, 1, Emax/Emin_bb, gams)

    r_M82 = R95_M82 / (Nmag_M82 * ff_1)
    r_M77_nb = R95_M77 / (Nmag_M77 * ff_nb)
    r_M77_bb = R95_M82 / (Nmag_M77 * ff_bb)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(gams, r_M82, label="M82 (nb, bb)", lw=4)
    ax.plot(gams, r_M77_nb, label="M77 (nb)", lw=3)
    ax.plot(gams, r_M77_bb, label="M77 (bb)", lw=2)

    ax.set_yscale('log')

    #ax.set_ylim(3e-4, 3)
    ax.set_xlim(-4., 0)
    ax.set_ylim(0.1, 1000)

    plt.grid(ls='--')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))

    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    ax.set_xlabel(r"$\gamma$", fontsize=16)
    ax.set_ylabel(r"$r_{95}\,\,({\rm magnetar}^{-1}~{\rm yr}^{-1})$", 
                  fontsize=16)

    plt.legend(fontsize=14)
    plt.show()
    return

    

def make_f_Eratio():
    Emin = 2e34
    Emax = 1e41  

    emin_r = 1
    emax_r = Emax / Emin

    gams = np.array([ 0, -1.5, -2.5, -3.5, -4.5 ])
    Efacs = 10**np.linspace(-1, 3, 1000) 

    fig = plt.figure()
    ax = fig.add_subplot(111)

    Ng = len(gams)
    lws = np.linspace(1.5, 3, Ng)
    #lcs = np.linspace(0, 0.5, Ng)

    for ii, gg in enumerate(gams):
        ff_ii = get_det_frac_arr2(Efacs, emin_r, emax_r, gg) 
        gstr = r"$\gamma = %+.1f$"%gg 
        #col = "%.3f" %lcs[ii]
        ax.plot(Efacs, ff_ii, lw=lws[ii], label=gstr)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylim(3e-4, 3)
    ax.set_xlim(0.1, 1e3)

    plt.grid(ls='--')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))

    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    ax.set_xlabel(r"$E / E_{\rm min}$", fontsize=16)
    ax.set_ylabel(r"$f(E)$", fontsize=16)

    plt.legend(fontsize=14)
    plt.show()
    return

    
