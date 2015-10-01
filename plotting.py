import numpy as _np
import matplotlib.pyplot as _plt
from corner import corner as _corner
from matplotlib.patches import Circle as _Circle
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap

from _settings import sigmas
from beam_func import offsets, beam_fwhm, beam_sep

# This is all just making my own colormap, one which is a single color but
# changes in alpha.  This is for plot_mcmc_beam_map.
image_color = (0, 0, 0.75, 1)
image_cmap = _LinearSegmentedColormap.from_list('image_cmap', [image_color, image_color], 256)
image_cmap._init()
alphas = _np.linspace(0, 0.8, image_cmap.N+3)
image_cmap._lut[:,-1] = alphas


# function to draw the circles that indicate where the beams are
def _draw_fwhm_circle(ax, xy, fwhm1, fwhm2, alpha=1.):
    outer_r = 0.5*max(fwhm1, fwhm2)
    inner_r = 0.5*min(fwhm1, fwhm2)
    ax.add_patch(_Circle(xy, outer_r, fc="0.95", lw=0, zorder=-100, alpha=alpha))
    ax.add_patch(_Circle(xy, inner_r, fc="0.90", lw=0, zorder=-99, alpha=alpha))


def _gaussian(x, loc, amp, sigma):
    return amp * _np.exp(-0.5*pow((x-loc)/sigma, 2))

def _make_data(offsets_index, chans, nsamp, snr_total, spec_index, xoff_arcmin, yoff_arcmin):
    nchan = len(chans)
    times = _np.linspace(0, 1, nsamp, endpoint=False)
    snrs = snr_vals(snr_total, chans, spec_index, xoff_arcmin, yoff_arcmin)
    dat = _np.random.normal(loc=0., scale=1., size=(nchan, nsamp))

    for ii in range(nchan):
        dat[ii] += _gaussian(times, 0.5, snrs[ii][offsets_index], 0.01)

    return times, dat


def plot_mcmc_results(sampled_mcmc, bins=20, smooth=None, truths=None):
    """
    Uses the 'corner' package to make a *sick* corner plot showing the
    projections of the parameters from this MCMC run in a multidimensional
    space.

    sampled_mcmc: The output of a pymc run.
    bins: The number of bins in the histogram.
    smooth: Higher number = more smooth (for the 2D plots).
    truths: List in [x, spec_index, y] order of known parameter values if applicable.
    """
    x_results = sampled_mcmc.x.trace()
    y_results = sampled_mcmc.y.trace()
    spec_index_results = sampled_mcmc.spec_index.trace()

    mcmc_all_results = _np.array([x_results, spec_index_results, y_results]).T
    corner_plot = _corner(mcmc_all_results, labels=["x", "spectral index", "y"], bins=bins, smooth=smooth, truths=truths)
    
    return corner_plot

def plot_mcmc_beam_map(sampled_mcmc, bins=50, true_xy=None):
    x_results = sampled_mcmc.x.trace()
    y_results = sampled_mcmc.y.trace()
    spec_index_results = sampled_mcmc.spec_index.trace()

    x_bins = _np.linspace(-beam_sep, beam_sep, bins+1)
    y_bins = _np.linspace(-beam_sep, beam_sep, bins+1)
    xy_hist, x_edges, y_edges = _np.histogram2d(x_results, y_results,\
      bins=(x_bins, y_bins))

    max2min = _np.sort(xy_hist.flatten())[::-1]
    cumul = _np.cumsum(max2min/_np.sum(max2min))
    val68, val95, val99 = max2min[_np.searchsorted(cumul,\
      [sigmas[1], sigmas[2], sigmas[3]])]

    fig = _plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    for ii in range(len(offsets)):
        _draw_fwhm_circle(ax, offsets[ii], beam_fwhm(400), beam_fwhm(800))
    prob_plot = ax.pcolorfast(x_bins, y_bins, xy_hist.T, cmap=image_cmap)
    ax.contour(0.5*(x_bins[:-1]+x_bins[1:]), 0.5*(y_bins[:-1]+y_bins[1:]),\
      xy_hist.T, (val68, val95, val99), colors='black',\
      linewidths=(1.5, 0.75, 0.25))
    
    if true_xy is not None:
        ax.plot(true_xy[0], true_xy[1], 'o', color='red')
    #ax.plot(pos[0], pos[1], '+', ms=10, color='white')
    #ax.plot(pos[0], pos[1], 'x', ms=10, color='black')
    
    ax.set_xlim(-beam_sep, beam_sep)
    ax.set_ylim(-beam_sep, beam_sep)
    ax.set_xlabel("arcmin")
    ax.set_ylabel("arcmin")
    
    return fig

def plot_mcmc_traces(sampled_mcmc, true_xy=None, true_spec_index=None):
    x_results = sampled_mcmc.x.trace()
    y_results = sampled_mcmc.y.trace()
    spec_index_results = sampled_mcmc.spec_index.trace()
    
    fig = _plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(311)
    ax1.plot(x_results, '.', color=(0,0,0,0.05))
    if true_xy is not None:
        ax1.plot([0, len(x_results)], [true_xy[0], true_xy[0]], color='red', ls='dashed', lw=2)
    ax1.set_ylabel('x')
    ax1.set_xlim(0, len(x_results))
    ax1.set_xticks([])

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(y_results, '.', color=(0,0,0,0.05))
    if true_xy is not None:
        ax2.plot([0, len(y_results)], [true_xy[1], true_xy[1]], color='red', ls='dashed', lw=2)
    ax2.set_ylabel('y')
    ax2.set_xlim(0, len(y_results))

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(spec_index_results, '.', color=(0,0,0,0.5))
    if true_spec_index is not None:
        ax3.plot([0, len(spec_index_results)], [true_spec_index, true_spec_index], color='red', ls='dashed', lw=2)
    ax3.set_ylabel('spectral index')
    ax3.set_xlim(0, len(spec_index_results))
    
    return fig


def plot9waterfalls(snr_total, chans, nsamp, x, y, spec_index):
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
            
        times, dat = _make_data(ii, chans, nsamp, snr_total, spec_index, x, y)
        ax = _plt.subplot2grid((3,3), (y_index, x_index))
        ax.pcolormesh(times, chans, dat, cmap='bone_r')
        ax.axis('off')

    return fig
