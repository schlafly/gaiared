gaiared
=======

Code to fit photometry & parallaxes as a function of spectroscopic parameters, to derive:
1. the extinction curve
2. its variation
3. the intrinsic absolute magnitudes of stars as a function of spectroscopic parameters

The input is a list of stars' photometry, parallaxes, and associated uncertainties, together with their spectroscopic parameters and uncertainties.  These are these fit as a big nonlinear minimization, trying to model the observed photometry and parallaxes as polynomials in the spectroscopic parameters, plus multiple extinction components (total column, R(V)), plus distance.

Typical usage is something like:
```
res = gaiared.fit_model(mag, dmag, parallax, dparallax, teff, logz, logg, extprior, extpriorsig, nextcomp=2, fitgrav=True, order=4, logg_prior_mean=logg, logg_prior_sigma=dlogg)
intmagpar, extcurvepar, mus, loggs = gaiared.unwrap_param(res, nband, nintmagpar, len(parallax), 2, fitgrav=True)
``` 
though the code is in an exploratory state.  In the above example, the gravities are not taken as fixed, and are instead fit (fitgrav=True).  In this case, they require priors from the spectroscopy (logg_prior_mean, logg_prior_sigma).  Some kind of extinction prior is required to remove normalization and zero point degeneracies.

A file merging PS1, 2MASS, WISE, and Gaia information for stars with APOGEE spectroscopy is available at <http://faun.rc.fas.harvard.edu/eschlafly/apored/misc/gaiaredxmatch.fits>.  That should contain all observations necessary for getting good measurements of the extinction curve, its variation, and constraining how stars' absolute magnitudes depend on their APOGEE spectroscopic parameters.

