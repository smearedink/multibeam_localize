import numpy as _np
import pylab as _plt
import pymc as _pymc
from corner import corner as _corner

from beam_func import beam_signal, beam_sep, offsets, centre_beam_index

import plotting

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


def locate_mcmc(x, y, spec_index, snr_centre_on_axis, chans, nsample=100000, burn=10000, add_error=False, x_lims=(-beam_sep, beam_sep), y_lims=(-beam_sep, beam_sep), spec_index_lims=(-10., 10.)):
    if _np.abs(x) > 0.5*beam_sep or _np.abs(y) > 0.5*beam_sep:
        print "WARNING: You are considering an event location outside the centre beam,"              " which will give results that might not make sense."
    
    mcmc_inputs = {}
    mcmc_inputs['x'] = _pymc.Uniform('x', x_lims[0], x_lims[1], value=0.)
    mcmc_inputs['y'] = _pymc.Uniform('y', y_lims[0], y_lims[1], value=0.)
    mcmc_inputs['spec_index'] = _pymc.Uniform('spec_index', spec_index_lims[0], spec_index_lims[1], value=0.)

    @_pymc.deterministic(plot=False)
    def _pymc_observed_relative_signals(x=mcmc_inputs['x'], y=mcmc_inputs['y'], spec_index=mcmc_inputs['spec_index'], chans=chans):
        return observed_relative_signals(x, y, spec_index, chans)
    
    # this is my "observed signal"
    sig = observed_relative_signals(x, y, spec_index, chans)
    
    off_axis_snr_scale = (_np.squeeze(beam_signal(x, y, chans))) * snr_centre_on_axis / _np.sqrt(_np.array(chans).size)
    err = 1./ (off_axis_snr_scale.sum(0) / _np.sqrt(_np.array(chans).size))
    
    if add_error:
        error = _np.random.normal(scale=err, size=sig.shape)
        sig += error
        
    mcmc_inputs['signals'] = _pymc.Normal('signals', mu=_pymc_observed_relative_signals, tau=1./err**2, value=sig, observed=True)
    
    R = _pymc.MCMC(input=mcmc_inputs)
    R.use_step_method(_pymc.AdaptiveMetropolis, [R.x, R.y, R.spec_index])
    R.sample(nsample, burn=burn)
    
    return R



