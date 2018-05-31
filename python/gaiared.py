import numpy
import pdb

# The straightforward model has some set of parameters describing the
# overall model and some set of parameters describing the individual
# stars.  The former consists of the relationship between intrinsic
# stellar parameters (teff, logz, logg) and the absolute magnitudes,
# as well as the relative amount of extinction in each band.  The
# latter is composed of the type of each star (teff, logz, logg), its
# distance, and numbers describing its extinction (E(B-V) and R(V) or
# some proxy for R(V)).  If the natural number of free parameters for
# each of these elements is allowed, the model has a number of perfect
# degeneracies surrounding the zero point, normalization, and
# separation of extinction into different components.  These are: 

# (1) One can rescale all of the extinctions and the reddening vector
# elements by a constant.  To fix this we could impose something like
# sum(extinction vector) == 1 (if not using a lookup table).  If using
# a lookup table, one could do something like dA(g) = 0, or maybe
# sum(dA) = 0.

# (2) If fitting multiple extinction components, one is really
# identifying an extinction plane, and any set of extinctions and
# extinction vectors that cover this plane will be equally good fits.
# In PCA, this degeneracy is broken by identifying one of the
# directions with the direction of greatest variance.  We need to fix
# two numbers, one for each vector (essentially the angle in this
# plane) in order to remove this degeneracy.  One could be fixed by
# demanding that the two vectors be orthogonal.  The other could be
# fixed by aligning within this plane close to parallel with a nominal
# extinction vector, but that's cumbersome.  One option in that
# direction would be to take dAg and dAr = 0 or something, sum(R^2) =
# 1, dRz = 0. Choose some intermediate band.

# (3) There is an overall zero point issue: one can move the magnitudes
# along the reddening vectors while changing the extinctions to leave
# no effect on the magnitudes.  If I insist that dR/dx has no effect on
# the colors when E -> 0, then maybe this just means fixing one magnitude
# as a function of teff, logz, logg.  Alternatively, if the sample
# include a set of zero-reddening objects, a prior on E for those would
# fix the zero point.  The latter is much more appealing, and also eliminates
# the normalization issue.  It only needs to be for a subset of stars and
# bands.

def compute_multinomials(order=3):
    """Computes relevant multinomial coefficients for use in modelmag."""
    
    xpows = []
    ypows = []
    zpows = []
    for xpow in range(order):
        for ypow in range(order):
            for zpow in range(order):
                if xpow+ypow+zpow > order:
                    continue
                xpows.append(xpow)
                ypows.append(ypow)
                zpows.append(zpow)
    return xpows, ypows, zpows
    

def modelmag(par, teff, logz, logg, nband=13, reset=False, gradient=False,
             fitgrav=False, order=3):
    """Returns intrinsic magnitudes according to some parameters describing
    the isochrones, and the temperature, metallicity, and gravity of
    the stars."""

    self = modelmag
    if reset or getattr(self, 'xpows', None) is None:
        self.xpows, self.ypows, self.zpows = compute_multinomials(order=order)

    x = (teff - 4600)/1000.
    y = logg - 2
    z = logz + 0.5
    par = par.reshape(nband, -1)
    if par.shape[-1] != len(self.xpows):
        raise ValueError('shape of par does not match expectations.')
    res = numpy.zeros((len(teff), nband), dtype='f8')
    if gradient:
        grad = getattr(self, 'grad', None)
        if (reset or (grad is None) or 
            (grad.shape != (len(teff), nband, par.shape[-1]))):
            grad = numpy.zeros((par.shape[-1], nband, len(teff)), 
                               dtype='f8').T
            self.grad = grad
        if fitgrav:
            gradg = getattr(self, 'gradg', None)
            if (reset or (gradg is None) or 
                (gradg.shape != (len(teff), nband))):
                gradg = numpy.zeros((nband, len(teff)), 
                                    dtype='f8').T
                self.gradg = gradg
    if gradient:
        grad[...] = 0.
        if fitgrav:
            gradg[...] = 0.
    for i, (xpow, ypow, zpow) in enumerate(zip(
            self.xpows, self.ypows, self.zpows)):
        xyz = x**xpow * y**ypow * z**zpow
        res += par[:, i].reshape(1, -1)*xyz.reshape(-1, 1)
        if gradient:
            grad[:, :, i] = xyz[:, None]
            if fitgrav:
                gradg += (
                    par[:, i].reshape(1, -1)*
                    (x**xpow*(ypow)*y**numpy.clip(ypow-1, 0, numpy.inf)*
                     z**zpow).reshape(-1, 1))
    # if we want to introduce T, logz, logg uncertainty, then
    # we also need to do derivatives with respect to these parameters.
    # that means xpow*(x**(xpow-1))*y**ypow*z**zpow
    if gradient:
        if fitgrav:
            return res, (grad, gradg)
        else:
            return res, grad
    else:
        return res


def fullmodel(intmagpar, extcurvepar, extinctions, mus, 
              teff, logz, logg, nband=13, gradient=False,
              extfun=None, fitgrav=False, order=3):
    absmag = modelmag(intmagpar, teff, logz, logg, nband=nband, 
                      gradient=gradient, fitgrav=fitgrav, order=order)
    if gradient:
        absmag, gradabsmag = absmag
    nextcomp = extcurvepar.shape[0]
    dumbfactor = 1
    # setting dumbfactor > 1 turns off R normalization
    # testing only!
    for i in range(nextcomp):
        fac = 1 if i < dumbfactor else numpy.sqrt(numpy.sum(extcurvepar[i]**2))
        absmag += extcurvepar[None, i, :]*extinctions[i][:, None]/fac
    if gradient:
        # by introducing a normalization factor into the
        # extinction curve parameters, we go from grad_{R_i} = x*(i==j)
        # to grad_{R_i} = x*((i==j)/|R| + R_i*R_j/|R|^3)
        # this gets big to store, so we store the R factor and the
        # extinctions separately
        nband = extcurvepar.shape[1]
        rirjfac = numpy.zeros((nextcomp, nband, nband), dtype='f8')
        for i in range(nextcomp):
            rirjfac[i] = numpy.eye(nband, nband)
            if i > dumbfactor-1:
                r0 = extcurvepar[i]
                fac = numpy.sqrt(numpy.sum(r0**2))
                rirjfac[i] /= fac
                rirjfac[i] -= r0.reshape(-1,1)*r0.reshape(1, -1)/fac**3
        gradextcurve = [extinctions, rirjfac]
        gradextinctions = extcurvepar.copy()
        for i in range(dumbfactor, nextcomp):
            fac = numpy.sqrt(numpy.sum(gradextinctions[i]**2.))
            gradextinctions[i] /= fac
    if extfun:
        nomextinctions = extfun(extinctions[0], teff, logz, logg, 
                                gradient=gradient)
        if gradient:
            nomextinctions, gradnomextinctions = nomextinctions
            gradextinctions = (gradextinctions[None, :, :]*
                               numpy.ones((nomextictions.shape[0], 1, 1)))
            gradextinctions[:, 0, :] += nomextinctions
    absmag += mus.reshape(-1, 1)
    if gradient:
        gradmu = numpy.array([1], dtype='f8').reshape(1, 1)
    if gradient:
        return absmag, gradabsmag, gradextcurve, gradextinctions, gradmu
    else:
        return absmag


def fullmodel_chi2(mag, dmag, parallax, dparallax, intmagpar, extcurvepar, 
                   extinctions, mus, teff, logz, logg, extinction_prior_mean,
                   extinction_prior_sigma,
                   nband=13, gradient=False, fitgrav=False, damp=3,
                   logg_prior_mean=None, logg_prior_sigma=None, order=3):
    tmodelmag = fullmodel(intmagpar, extcurvepar, extinctions, mus, 
                          teff, logz, logg, nband=nband, gradient=gradient,
                          fitgrav=fitgrav, order=order)
    if gradient:
        (tmodelmag, gradabsmag, gradextcurve, gradextinctions, 
         gradmu) = tmodelmag
        if fitgrav:
            gradabsmag, gradg = gradabsmag
    # the gradient bit of this is huge.
    # we only need it after taking the sum, though, so it's okay...
    chimag = damper((mag-tmodelmag)/dmag, damp)
    # dimensions: npar, nstar, nband
    if gradient:
        dfdchiodmag = damper_deriv((mag-tmodelmag)/dmag, damp)/dmag
        # needs to be multiplied into gradient correctly.
        nintmagpar = intmagpar.shape[0]*intmagpar.shape[1]
        nextcurvepar = extcurvepar.shape[0]*extcurvepar.shape[1]
        nextinctions = extinctions.shape[0]*extinctions.shape[1]
        nstar = mus.shape[0]
        npar = (nintmagpar + nextcurvepar + nextinctions + nstar)
        if fitgrav:
            npar += nstar
        # nstar <-> mu; needs to go to 4*nstar eventually to incorporate
        # teff, logz, logg
        delchi2 = numpy.zeros(npar, dtype='f8')
        i = 0
        delchi2[0:nintmagpar] = -2*numpy.einsum(
            'ij,ij,ijk->jk', chimag, dfdchiodmag, gradabsmag).reshape(-1)
        i += nintmagpar
        extcurveparstart, extcurveparend = i, i+nextcurvepar
        delchi2[i:i+nextcurvepar] = -2*numpy.einsum(
            'ij,ki,kjl->kl', chimag*dfdchiodmag, gradextcurve[0],
            gradextcurve[1]).reshape(-1)
        i += nextcurvepar
        extstart, extend = i, i+nextinctions
        delchi2[i:i+nextinctions] = -2*numpy.sum(chimag*dfdchiodmag*
            gradextinctions[:, None, :], axis=2).reshape(-1)
        i += nextinctions
        mustart, muend = i, i+nstar
        delchi2[i:i+nstar] = -2*numpy.sum(chimag*dfdchiodmag*gradmu, 
                                         axis=1)
        i += nstar
        loggparstart = i
    chiparallax = damper((parallax-mu_to_parallax(mus))/dparallax, damp)
    if gradient:
        dfdchiparallax = damper_deriv((parallax-mu_to_parallax(mus))/dparallax, 
                                      damp)
        gradparallax = -numpy.log(10)/5*100*10**(-mus/5)
        delchi2[mustart:muend] += (
            -2*chiparallax*dfdchiparallax*gradparallax/dparallax)
    chi2 = numpy.sum(chimag**2)+numpy.sum(chiparallax**2)
    if len(extinctions) > 0:
        extchi = (extinction_prior_mean-extinctions)/extinction_prior_sigma
        extchidamp = damper(extchi, damp)
        dfdchi = damper_deriv(extchi, damp)
        chi2 += numpy.sum(extchidamp**2)
        if gradient:
            delchi2[extstart:extend] += (
                -2*extchidamp*dfdchi/extinction_prior_sigma).reshape(-1)
    if len(extcurvepar) > 1:
        unit2normsigma = 0.1
        # chi = (1-numpy.sum(extcurvepar[1]**2))/unit2normsigma
        norm2 = numpy.sum(extcurvepar[1]**2.)
        chi = -numpy.log(numpy.sum(norm2))/unit2normsigma
        chi2 += chi**2
        if gradient:
            i = extcurveparstart+nband
            delchi2[i:i+nband] += -4*chi/unit2normsigma*extcurvepar[1]/norm2
        if len(extcurvepar) > 2:
            raise ValueError('Have not thought about normalization in '
                             'nextcomp > 2 case.')
    if fitgrav:
        chi = (logg_prior_mean-logg)/logg_prior_sigma
        chi2 += numpy.sum(chi**2)
        if gradient:
            delchi2[loggparstart:loggparstart+nstar] = -2*numpy.sum(
                chimag*dfdchiodmag*gradg, axis=1)
            delchi2[loggparstart:loggparstart+nstar] += (
                -2*chi/logg_prior_sigma)
    if gradient:
        return chi2, delchi2
    else:
        return chi2
        

def damper(chi, damp):
    """Pseudo-Huber loss function."""
    return 2*damp*numpy.sign(chi)*(numpy.sqrt(1+numpy.abs(chi)/damp)-1)
    # return chi/numpy.sqrt(1+numpy.abs(chi)/damp)


def damper_deriv(chi, damp):
    """Derivative of the pseudo-Huber loss function."""
    return 1./numpy.sqrt(1+numpy.abs(chi)/damp)


def mu_to_parallax(mu):
    return 10**(-mu/5.)*100


def parallax_to_mu(pi):
    return 5*numpy.log10(100/pi)


def make_bad_mock_data(nband=13, nstar=10000, nextcomp=2, order=3):
    pows = compute_multinomials(order=order)
    nintmagpar = len(pows[0])
    intmagpar = numpy.random.randn(nband, nintmagpar)
    extcurvepar = numpy.random.randn(nextcomp, nband)
    extcurvepar /= numpy.sqrt(numpy.sum(extcurvepar**2, axis=1)).reshape(-1, 1)
    extinctions = numpy.random.randn(nextcomp, nstar)
    mus = numpy.random.randn(nstar)+10
    teff = numpy.random.randn(nstar)*1000+4600
    logz = numpy.random.randn(nstar)
    logg = numpy.random.randn(nstar)
    tmodelmag = fullmodel(intmagpar, extcurvepar, extinctions, mus,
                          teff, logz, logg, nband=nband, order=order)
    tobsmag = tmodelmag + numpy.random.randn(*tmodelmag.shape)*0.01
    # 1% uncertainties... though without scales on anything else, pretty
    # meaningless.
    tmodelunc = numpy.ones_like(tmodelmag)*0.01
    tparallax = (mu_to_parallax(mus)+
                 numpy.random.randn(len(mus))*0.1)
    tparallaxunc = numpy.ones_like(tparallax)*0.1
    return (tobsmag, tmodelunc, tparallax, tparallaxunc,
            teff, logz, logg, mus, extinctions, intmagpar, extcurvepar)


def wrap_param(intmagpar, extcurvepar, extinctions, mus, 
               nband, nintmagpar, nstar, grav=None):
    if grav is None:
        grav = []
    return numpy.concatenate(
        [intmagpar.ravel(), extcurvepar.ravel(), extinctions.ravel(), mus, 
         grav])


def unwrap_param(param, nband, nintmagpar, nstar, nextcomp, fitgrav=False):
    i = 0
    intmagpar = param[i:i+nband*nintmagpar].reshape(nband, nintmagpar)
    i += nband*nintmagpar
    extcurvepar = param[i:i+nband*nextcomp].reshape(nextcomp, nband)
    i += nband*nextcomp
    extinctions = param[i:i+nstar*nextcomp].reshape(nextcomp, nstar)
    i += nextcomp*nstar
    mus = param[i:i+nstar]
    i += nstar
    res = intmagpar, extcurvepar, extinctions, mus
    if fitgrav:
        grav = param[i:i+nstar]
        i += nstar
        res = res + (grav,)
    if i != len(param):
        raise ValueError('shapes do not match!')
    return res


def fit_model(mag, dmag, parallax, dparallax, teff, logz, logg, extprior,
              extpriorsig, nextcomp=2, guess=None, fitgrav=False,
              logg_prior_mean=None, logg_prior_sigma=None, order=3):
    nstar = mag.shape[0]
    nband = mag.shape[1]
    nintmagpar = len(compute_multinomials(order=order)[0])
    print('Should cut out YSOs somewhere')
    if guess is None:
        # intmagparguess = numpy.zeros((nband, nintmagpar), dtype='f8')
        intmagparguess = numpy.random.randn(nband, nintmagpar)
        extcurveparguess = numpy.random.randn(nextcomp, nband)
        if nband == 12:
            intmagparguess[:, 0] = numpy.array([
                    0.2, -0.5, -0.8, -0.9, -1.0, -2.0, -2.6, -2.7, -2.7, -2.6,
                    -0.1, -1.1], dtype='f8')
            # just get the constant term right.
            if nextcomp > 0:
                # probably: grizYJHK12BR
                extcurveparguess[0, :] = numpy.array(
                    [3.5, 2.7, 2.0, 1.6, 1.3, 0.8, 0.5, 0.3, 0.2, 0.2, 
                     3.5, 2.0])
        extinctionsguess = extprior.copy()
        musguess = parallax_to_mu(numpy.clip(parallax, 0.01, numpy.inf))
        if fitgrav:
            gravguess = logg
        else:
            gravguess = None
    else:
        if fitgrav:
            gravguess = guess[-1]
            guess = guess[:-1]
        else:
            gravguess = None
        intmagparguess, extcurveparguess, extinctionsguess, musguess = guess
    guess = wrap_param(intmagparguess, extcurveparguess,
                       extinctionsguess, musguess, nband, nintmagpar, nstar,
                       grav=gravguess)
    from scipy.optimize import fmin_l_bfgs_b, fmin_cg
    def chi2_wrapper(param):
        upar = unwrap_param(param, nband, nintmagpar, nstar, nextcomp, 
                            fitgrav=fitgrav)
        intmagpar, extcurvepar, extinctions, mus = upar[0:4]
        if fitgrav:
            tlogg = upar[4]
        else:
            tlogg = logg
        chi2, grad = fullmodel_chi2(mag, dmag, parallax, dparallax,
                                    intmagpar, extcurvepar, extinctions,
                                    mus, teff, logz, tlogg, extprior,
                                    extpriorsig, nband=nband, 
                                    gradient=True, fitgrav=fitgrav,
                                    logg_prior_mean=logg_prior_mean,
                                    logg_prior_sigma=logg_prior_sigma,
                                    order=order)
        return chi2, grad
    res = fmin_l_bfgs_b(chi2_wrapper, guess, m=100, iprint=10,
                        maxiter=300000, maxfun=300000)
    # res = fmin_cg(chi2_wrapper, guess, fprime=grad_wrapper, maxiter=10000)
    return res

        
def numerical_gradients(chi2gradfun, param):
    chi2, grad = chi2gradfun(param)
    dq = 1e-5
    res = []
    for i in range(len(param)):
        newparam = param + dq*(numpy.arange(len(param)) == i)
        chi20 = chi2gradfun(newparam)[0]
        res.append((chi20-chi2)/dq)
    return numpy.array(res)


def mask(ob):
    # what do we need?
    # good APOGEE parameters
    # ... that's about it?
    # that just duplicates existing selection.
    badflags1mask = 0
    badflags2mask = 0
    badflags3mask = 0
    badflags1 = [7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                 25, 26, 27, 28, 29, 30]
    badflags2 = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22 ]
    badflags3 = [8, 24, 10, 26] # chi2 warn, rotation warn, bad
    for f in badflags1:
        badflags1mask = badflags1mask | (2**f)
    for f in badflags2:
        badflags2mask = badflags2mask | (2**f)
    for f in badflags3:
        badflags3mask = badflags3mask | (2**f)
    
    good = (((ob['apogee_target1'] & badflags1mask) == 0) &
            ((ob['apogee_target2'] & badflags2mask) == 0) &
            ((ob['aspcapflag'] & badflags3mask) == 0))
    if 'apogee2_target3' in ob.dtype.names:
        inars = (ob['apogee2_target3'] & 2**20) != 0
        # any other program should be excluded
        badflags4 = [0, 1, 2, 3, 4, 5, 8]
        ancillarymask = 0
        for f in badflags4:
            ancillarymask = ancillarymask | (2**f)
        good = good & (inars | ((ob['apogee_target3'] & ancillarymask) == 0))
    good = (good & 
            (ob['teff'] > 3500) & (ob['logg'] > -5) & (ob['m_h'] > -6))
    # PS1 is probably less blended than 2MASS/WISE, so mask all if extended
    # in PS1
    apexcessps = (ob['median_ap']/numpy.clip(
            ob['median'], 1.e-9, numpy.inf)) - 1
    m = (numpy.abs(apexcessps) > 0.2) & (ob['nmag_ok'] > 0)
    good = good & ~numpy.any(m, axis=1)
    good = good & (ob['posstdev_ok'] < 0.1)  # no-op for no PS1 match
    return good


def xmatch_to_good_output(xmatch):
    # xmatch file contains all of the columns from all of the
    # APOGEE objects, matched to Gaia/PS1/WISE
    # we want to select this down to the good subsample
    # and also pull out the relevant rows
    m = mask(xmatch)
    xmatch = xmatch[m]
    mag = numpy.zeros((len(xmatch), 12), dtype='f8')
    unc = numpy.zeros_like(mag)
    magps = -2.5*numpy.log10(numpy.clip(xmatch['median'], 
                                        1e-30, numpy.inf))
    uncps = 2.5/numpy.log(10)*xmatch['err']/numpy.clip(xmatch['median'], 
                                                       1e-30, numpy.inf)
    uncps[uncps == 0] = numpy.inf
    uncps = numpy.sqrt(uncps**2. + 0.01**2.)
    m = uncps == 0.
    uncps[m] = numpy.inf
    mag[:, :5] = magps
    unc[:, :5] = uncps
    saturation_ps1 = [14.0, 14.4, 14.4, 13.8, 13.] 
    for i, bright_mag in enumerate(saturation_ps1):
        m = mag[:, i] < bright_mag
        unc[m, i] = numpy.inf
    for i, filt in enumerate('jhk'):
        mag[:, 5+i] = xmatch[filt]
        unc[:, 5+i] = xmatch[filt+'_err']
    for i, filt in enumerate(['w1mpro', 'w2mpro']):
        m = numpy.array([flags[0] != '0' for flags in xmatch['cc_flags']])
        mag[:, 8+i] = xmatch[filt]
        unc[:, 8+i] = xmatch[filt[0:2]+'sig'+filt[2:]]
        unc[m, 8+i] = numpy.inf
    for i, filt in enumerate(['bp', 'rp']):
        fstr = 'phot_'+filt+'_mean_flux'
        tflux, tunc = xmatch[fstr].copy(), xmatch[fstr+'_error'].copy()
        tmag = xmatch[fstr[:-4]+'mag']
        m = numpy.isnan(tflux) | numpy.isnan(tunc) | numpy.isnan(tmag)
        tflux[m] = 0
        tmag[m] = 0
        tunc[m] = numpy.inf
        tunc = (2.5/numpy.log(10)*tunc/numpy.clip(tflux, 1e-30, numpy.inf))
        tunc = numpy.sqrt(tunc**2.+0.005**2.)
        mag[:, 10+i] = tmag
        unc[:, 10+i] = tunc
        
    if numpy.any(unc <= 0):
        m = unc <= 0
        unc[m] = numpy.inf
    res = numpy.zeros(len(xmatch), 
                      dtype=[('ra', 'f8'), ('dec', 'f8'), 
                             ('l', 'f8'), ('b', 'f8'),
                             ('mag', 'f8', 12), ('dmag', 'f8', 12),
                             ('parallax', 'f8'), ('dparallax', 'f8'),
                             ('teff', 'f8'), ('dteff', 'f8'),
                             ('logz', 'f8'), ('dlogz', 'f8'),
                             ('logg', 'f8'), ('dlogg', 'f8')])
    res['mag'] = mag
    res['dmag'] = unc
    res['parallax'] = xmatch['parallax']
    res['dparallax'] = xmatch['parallax_error']
    m = (~numpy.isfinite(xmatch['parallax']) |
          ~numpy.isfinite(xmatch['parallax_error']))
    res['parallax'][m] = 0.
    res['dparallax'][m] = numpy.inf
    m = res['dparallax'] < 0
    res['parallax'][m] = 0.
    res['dparallax'][m] = numpy.inf
    res['teff'] = xmatch['teff']
    res['logz'] = xmatch['m_h']
    res['logg'] = xmatch['logg']
    res['dteff'] = xmatch['teff_err']
    res['dlogz'] = xmatch['m_h_err']
    res['dlogg'] = xmatch['logg_err']
    res['ra'] = xmatch['ra']
    res['dec'] = xmatch['dec']
    res['l'], res['b'] = xmatch['glon'], xmatch['glat']
    return res, xmatch


def extinction_prior(ob):
    import dust
    avfac = 2.742  # SF11, R(V) = 3.1 for Landolt A(V)
    # any star with an extinction prior gets an E*R(V) prior,
    # mean of 0.2*extinction prior.
    # nearby stars get an extinction prior of 0.
    # there really aren't many nearby stars (<100 pc; these are APOGEE
    # giants, so no surprise).  So let's just do |b| > 30, D > 1 kpc.
    m = (numpy.abs(ob['b']) > 30) & (ob['parallax'] < 1)
    extprior = numpy.zeros((2, len(ob)), dtype='f8')
    extpriorsig = numpy.zeros((2, len(ob)), dtype='f8')
    extprior[0, m] = dust.getval(ob['l'][m], ob['b'][m], map='sfd')*avfac
    extpriorsig[0, m] = 0.1*extprior[0, m]
    extpriorsig[0, m] = numpy.sqrt(extpriorsig[0, m]**2. + (0.01*avfac)**2.)
    extpriorsig[0, ~m] = 30
    extpriorsig[1, m] = extpriorsig[0, m]
    extpriorsig[1, ~m] = 3
    # really rough guesses!
    return extprior, extpriorsig

