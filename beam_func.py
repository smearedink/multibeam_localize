import numpy as _np


speed_of_light = 299792458.
D_cyl = 20.
sincsq_halfmax = 0.44295



# beams are uniformly gridded in sin(theta) which I take to mean 256 beams from -1 to 1,
# which I take to mean at zenith arcsin(2/256) is an acceptable approximation to beam separation
beam_sep = _np.arcsin(2./256.) * 60. * 180/_np.pi
offsets = beam_sep * _np.array(_np.meshgrid([-1.,0.,1.], [-1.,0.,1.])).T.reshape(-1, 2)
centre_beam_index = 4



def beam_fwhm(freq):
    wavelength = speed_of_light/(_np.array(freq)*1.e6)
    return 60.*180./_np.pi * _np.arcsin(wavelength/100.)

def beam_signal(xoff_arcmin, yoff_arcmin, freq=400.):
    beam_fwhm_arcmin = beam_fwhm(freq)
    offset_arcmin = _np.sqrt(xoff_arcmin**2 + yoff_arcmin**2)
    return _np.sinc(sincsq_halfmax*_np.array(offset_arcmin)[..., _np.newaxis]/(0.5*_np.array(beam_fwhm_arcmin)[_np.newaxis, ...]))**2



