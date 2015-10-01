import numpy as _np
import matplotlib.pyplot as _plt
import pymc as _pymc
from corner import corner as _corner

from beam_func import beam_signal, beam_sep, offsets, centre_beam_index

import plotting

def get_chan_centres(nchan, band=(400., 800.)):
    chans = _np.linspace(band[0], band[1], nchan, endpoint=False)
    return chans + (band[1]-band[0])/(2.*nchan)
    
def get_chan_edges(nchan, band=(400., 800.)):
    chans = _np.linspace(band[0], band[1], nchan+1, endpoint=True)
    return chans

def all_beam_signals(x, y, freq=400.):
    return _np.squeeze(beam_signal(_np.array(x)[..., _np.newaxis]-offsets[:,0][_np.newaxis, ...],
                                  _np.array(y)[..., _np.newaxis]-offsets[:,1][_np.newaxis, ...],
                                  _np.squeeze(_np.array([freq]))))


# a function that takes total S/N as well as channels and spectral index and outputs
# S/N per channel (now includes beam offsets!!!!)
def snr_vals(snr_total, chans, spec_index, xoff_arcmin=0, yoff_arcmin=0):
    """
    snr_total is the on-axis (peak) S/N one gets by summing up all channels
    """
    if _np.isscalar(chans):
        chans = _np.array([chans])
    vals = _np.zeros_like(chans)
    vals[0] = 1.
    for ii in xrange(1, len(chans)):
        vals[ii] = vals[0]*(chans[ii]/chans[0])**spec_index
    signal_total = vals.sum()
    noise_total = _np.sqrt(len(chans))
    snr = signal_total / noise_total
    on_axis_vals = vals * snr_total / snr
    return (on_axis_vals.T * all_beam_signals(xoff_arcmin, yoff_arcmin, chans)).T


# since these are relative signals, the centre beam is always 1, so the error is 1/snr_centre
def observed_relative_signals(x, y, spec_index, chans):
    nchan = _np.array(chans).size
    # signals per beam with on-axis freq-summed centre beam s/n = 1 (arbitrarily)
    sig = snr_vals(1., chans, spec_index, x, y)
    # sum over channels
    relative_signals = _np.sum(sig, axis=0) / _np.sqrt(nchan)
    scale_factor = 1./relative_signals[centre_beam_index]
    scaled_signals = sig[_np.newaxis, ...] * scale_factor
    
    return _np.squeeze(scaled_signals)


def locate_mcmc(x, y, spec_index, snr_centre_on_axis, nchan, band=(400., 800.), nsample=100000, burn=10000, add_error=False, x_lims=(-beam_sep, beam_sep), y_lims=(-beam_sep, beam_sep), spec_index_lims=(-10., 10.)):
    if _np.abs(x) > 0.5*beam_sep or _np.abs(y) > 0.5*beam_sep:
        print "WARNING: You are considering an event location outside the centre beam,"              " which will give results that might not make sense."
    
    chans = get_chan_centres(nchan, band)
    
    mcmc_inputs = {}
    mcmc_inputs['x'] = _pymc.Uniform('x', x_lims[0], x_lims[1], value=0.)
    mcmc_inputs['y'] = _pymc.Uniform('y', y_lims[0], y_lims[1], value=0.)
    mcmc_inputs['spec_index'] = _pymc.Uniform('spec_index', spec_index_lims[0], spec_index_lims[1], value=0.)

    @_pymc.deterministic(plot=False)
    def _pymc_observed_relative_signals(x=mcmc_inputs['x'], y=mcmc_inputs['y'], spec_index=mcmc_inputs['spec_index'], chans=chans):
        return observed_relative_signals(x, y, spec_index, chans)
    
    # this is my "observed signal"
    sig = observed_relative_signals(x, y, spec_index, chans)
    
    off_axis_snr_scale = (_np.squeeze(beam_signal(x, y, chans))) * snr_centre_on_axis / _np.sqrt(nchan)
    err = 1./ (off_axis_snr_scale.sum(0) / _np.sqrt(nchan))
    
    if add_error:
        error = _np.random.normal(scale=err, size=sig.shape)
        sig += error
        
    mcmc_inputs['signals'] = _pymc.Normal('signals', mu=_pymc_observed_relative_signals, tau=1./err**2, value=sig, observed=True)
    
    R = _pymc.MCMC(input=mcmc_inputs)
    R.use_step_method(_pymc.AdaptiveMetropolis, [R.x, R.y, R.spec_index])
    R.sample(nsample, burn=burn)
    
    return R


def _gaussian(x, loc, amp, sigma):
    return amp * _np.exp(-0.5*pow((x-loc)/sigma, 2))

def _make_data(offsets_index, nchan, nsamp, snr_total, spec_index, xoff_arcmin, yoff_arcmin, band=(400., 800.)):
    chans = get_chan_centres(nchan, band)
    
    times = _np.linspace(0, 1, nsamp+1, endpoint=True)
    snrs = snr_vals(snr_total, chans, spec_index, xoff_arcmin, yoff_arcmin)
    dat = _np.random.normal(loc=0., scale=1., size=(nchan, nsamp))

    for ii in range(nchan):
        dat[ii] += _gaussian(times[:-1], 0.5, snrs[ii][offsets_index], 0.01)

    return times, get_chan_edges(nchan, band), dat

def plot_9_signals(snr_total, nchan, nsamp, x, y, spec_index, band=(400., 800.), waterfall=True):
    fig = _plt.figure(figsize=(6,6))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for ii in range(9):
        this_x, this_y = offsets[ii]
        if this_x < -0.001:
            x_index = 0
        elif this_x > 0.001:
            x_index = 2
        else:
            x_index = 1
            
        if this_y < -0.001:
            y_index = 2
        elif this_y > 0.001:
            y_index = 0
        else:
            y_index = 1
            
        plot_times, plot_chans, dat = _make_data(ii, nchan, nsamp, snr_total, spec_index, x, y, band)
        ax = _plt.subplot2grid((3,3), (y_index, x_index))
        if waterfall:
            ax.pcolormesh(plot_times, plot_chans, dat, cmap='bone_r')
        else:
            ax.plot(plot_times[:-1], dat.sum(axis=0)/_np.sqrt(nchan), color='black')
        ax.axis('off')

    return fig