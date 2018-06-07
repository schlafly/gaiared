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

parnames = ['teff', 'logz', 'logg']

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
    

def modelmag(par, starpar, nband=13, reset=False, gradient=False,
             fitpar=[], order=3):
    """Returns intrinsic magnitudes according to some parameters describing
    the isochrones, and the temperature, metallicity, and gravity of
    the stars."""

    self = modelmag
    if reset or getattr(self, 'xpows', None) is None:
        self.xpows, self.ypows, self.zpows = compute_multinomials(order=order)

    x = (starpar['teff'] - 4600)/1000.
    y = starpar['logg'] - 2
    z = starpar['logz'] + 0.5
    par = par.reshape(nband, -1)
    if par.shape[-1] != len(self.xpows):
        raise ValueError('shape of par does not match expectations.')
    nstar = len(starpar['teff'])
    res = numpy.zeros((nstar, nband), dtype='f8')
    if gradient:
        grad = getattr(self, 'grad', None)
        if (reset or (grad is None) or 
            (grad.shape != (nstar, nband, par.shape[-1]))):
            grad = numpy.zeros((par.shape[-1], nband, nstar), 
                               dtype='f8').T
            self.grad = grad
        for parname in fitpar:
            gradpar = getattr(self, 'grad%s' % parname, None)
            if (reset or (gradpar is None) or 
                (gradpar.shape != (nband, nstar))):
                gradpar = numpy.zeros((nband, nstar), dtype='f8').T
                setattr(self, 'grad%s' % parname, gradpar)
    if gradient:
        grad[...] = 0.
        for parname in fitpar:
            gradpar = getattr(self, 'grad%s' % parname)
            gradpar[...] = 0.
    for i, (xpow, ypow, zpow) in enumerate(zip(
            self.xpows, self.ypows, self.zpows)):
        xyz = x**xpow * y**ypow * z**zpow
        res += par[:, i].reshape(1, -1)*xyz.reshape(-1, 1)
        if gradient:
            grad[:, :, i] = xyz[:, None]
            if 'teff' in fitpar:
                self.gradteff += (
                    par[:, i].reshape(1, -1)*
                    (xpow*x**numpy.clip(xpow-1, 0, numpy.inf)*y**ypow*
                     z**zpow).reshape(-1, 1))
            if 'logg' in fitpar:
                self.gradlogg += (
                    par[:, i].reshape(1, -1)*
                    (x**xpow*ypow*y**numpy.clip(ypow-1, 0, numpy.inf)*
                     z**zpow).reshape(-1, 1))
            if 'logz' in fitpar:
                self.gradlogz += (
                    par[:, i].reshape(1, -1)*
                    (x**xpow*y**ypow*
                     zpow*z**numpy.clip(zpow-1, 0, numpy.inf)).reshape(-1, 1))
    if gradient:
        gradout = {'int': grad}
        for parname in fitpar:
            gradout[parname] = getattr(self, 'grad%s' % parname)
        return res, gradout
    else:
        return res


def fullmodel(intmagpar, extcurvepar, starpar,
              nband=13, gradient=False,
              extfun=None, fitpar=[], order=3):
    absmag = modelmag(intmagpar, starpar, nband=nband, 
                      gradient=gradient, fitpar=fitpar, order=order)
    if gradient:
        absmag, gradabsmag = absmag
    nextcomp = extcurvepar.shape[0]
    normstart = 1
    gradstarpar = {}
    for i in range(nextcomp):
        fac = 1 if i < normstart else numpy.sqrt(numpy.sum(extcurvepar[i]**2))
        absmag += extcurvepar[None, i, :]*starpar['extinction'][:, i, None]/fac
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
            if i >= normstart:
                r0 = extcurvepar[i]
                fac = numpy.sqrt(numpy.sum(r0**2))
                rirjfac[i] /= fac
                rirjfac[i] -= r0.reshape(-1,1)*r0.reshape(1, -1)/fac**3
        gradextcurve = [starpar['extinction'].T, rirjfac]
        gradstarpar['extinction'] = extcurvepar.copy()
        for i in range(normstart, nextcomp):
            fac = numpy.sqrt(numpy.sum(gradstarpar['extinction'][i]**2.))
            gradstarpar['extinction'][i] /= fac
    if extfun:
        # this block needs to be looked at again.
        # the nominal extinctions need to be added to the absolute magnitudes.
        # check what's going on with the gradient calculations
        # this is clearly not done yet!
        raise NotImplementedError('this is not done yet!')
        nomextinctions = extfun(starpar['extinction'][0], starpar['teff'],
                                starpar['logz'], starpar['logg'],
                                gradient=gradient)
        if gradient:
            nomextinctions, gradnomextinctions = nomextinctions
            gradstarpar['extinction'] = (
                gradstarpar['extinction'][None, :, :] *
                numpy.ones((nomextinctions.shape[0], 1, 1)))
            gradstarpar['extinction'][:, 0, :] += gradnomextinctions
    absmag += starpar['mu'].reshape(-1, 1)
    if gradient:
        gradstarpar['mu'] = numpy.array([1], dtype='f8').reshape(1, 1)
    if gradient:
        return absmag, gradabsmag, gradextcurve, gradstarpar
    else:
        return absmag


def fullmodel_chi2(star, intmagpar, extcurvepar, starpar,
                   nband=13, gradient=False, fitpar=[], damp=3,
                   order=3, apply_extcurvenormprior=True):
    for parname in parnames:
        if parname not in starpar:
            starpar[parname] = star[parname]
    tmodelmag = fullmodel(intmagpar, extcurvepar, starpar, 
                          nband=nband, gradient=gradient,
                          fitpar=fitpar, order=order)
    if gradient:
        (tmodelmag, gradabsmag, gradextcurve, gradstarpar) = tmodelmag
    chimag = damper((star['mag']-tmodelmag)/star['dmag'], damp)
    # dimensions: npar, nstar, nband
    if gradient:
        dfdchiodmag = damper_deriv((star['mag']-tmodelmag)/star['dmag'], 
                                   damp)/star['dmag']
        # needs to be multiplied into gradient correctly.
        nintmagpar = intmagpar.shape[0]*intmagpar.shape[1]
        nextcurvepar = extcurvepar.shape[0]*extcurvepar.shape[1]
        nextinctions = (starpar['extinction'].shape[0] *
                        starpar['extinction'].shape[1])
        nstar = starpar['mu'].shape[0]
        npar = (nintmagpar + nextcurvepar + nextinctions + 
                nstar*(len(fitpar)+1))
        delchi2 = numpy.zeros(npar, dtype='f8')
        i = 0
        delchi2[0:nintmagpar] = -2*numpy.einsum(
            'ij,ij,ijk->jk', chimag, dfdchiodmag, 
            gradabsmag['int']).reshape(-1)
        i += nintmagpar
        extcurveparstart, extcurveparend = i, i+nextcurvepar
        delchi2[i:i+nextcurvepar] = -2*numpy.einsum(
            'ij,ki,kjl->kl', chimag*dfdchiodmag, gradextcurve[0],
            gradextcurve[1]).reshape(-1)
        i += nextcurvepar
        extstart, extend = i, i+nextinctions
        delchi2[i:i+nextinctions] = -2*numpy.sum(chimag*dfdchiodmag*
            gradstarpar['extinction'][:, None, :], axis=2).reshape(-1)
        i += nextinctions
        mustart, muend = i, i+nstar
        delchi2[i:i+nstar] = -2*numpy.sum(chimag*dfdchiodmag*gradstarpar['mu'],
                                          axis=1)
        i += nstar
        specparstart = i
    chiparallax = damper((star['parallax']-mu_to_parallax(starpar['mu']))/
                         star['dparallax'], damp)
    if gradient:
        dfdchiparallax = damper_deriv(
            (star['parallax']-mu_to_parallax(starpar['mu']))/star['dparallax'],
            damp)
        gradparallax = -numpy.log(10)/5*100*10**(-starpar['mu']/5)
        delchi2[mustart:muend] += (
            -2*chiparallax*dfdchiparallax*gradparallax/star['dparallax'])
    chi2 = numpy.sum(chimag**2)+numpy.sum(chiparallax**2)
    if starpar['extinction'].shape[1] > 0:
        extchi = ((star['extprior']-starpar['extinction'])/
                  star['extpriorsig'])
        extchidamp = damper(extchi, damp)
        dfdchi = damper_deriv(extchi, damp)
        chi2 += numpy.sum(extchidamp**2)
        if gradient:
            delchi2[extstart:extend] += (
                -2*extchidamp*dfdchi/star['extpriorsig']).T.reshape(-1)
    if apply_extcurvenormprior and (len(extcurvepar) > 1):
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
    i = specparstart
    for parname in parnames:
        if parname in fitpar:
            chi = (star[parname]-starpar[parname])/star['d%s' % parname]
            chi2 += numpy.sum(chi**2.)
            if gradient:
                delchi2[i:i+nstar] = -2*numpy.sum(
                    chimag*dfdchiodmag*gradabsmag[parname], axis=1)
                delchi2[i:i+nstar] += -2*chi/star['d%s' % parname]
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
    extinctions = numpy.random.randn(nstar, nextcomp)
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


def wrap_param(intmagpar, extcurvepar, starpar, 
               nband, nintmagpar, nstar, fitpar=[]):
    res = [intmagpar.ravel(), extcurvepar.ravel(),
           starpar['extinction'].T.ravel(), starpar['mu']]
    for parname in parnames:
        if parname in fitpar:
            res += [starpar[parname].ravel()]
    return numpy.concatenate(res)


def unwrap_param(param, nband, nintmagpar, nstar, nextcomp, fitpar=[]):
    i = 0
    intmagpar = param[i:i+nband*nintmagpar].reshape(nband, nintmagpar)
    i += nband*nintmagpar
    extcurvepar = param[i:i+nband*nextcomp].reshape(nextcomp, nband)
    i += nband*nextcomp
    starpar = {}
    starpar['extinction'] = (
        param[i:i+nstar*nextcomp].reshape(nextcomp, nstar).T)
    i += nextcomp*nstar
    starpar['mu'] = param[i:i+nstar]
    i += nstar
    res = intmagpar, extcurvepar, starpar
    for parname in parnames:
        if parname in fitpar:
            starpar[parname] = param[i:i+nstar]
            i += nstar
    if i != len(param):
        raise ValueError('shapes do not match!')
    return res


# starpar must have:
# mag, dmag, parallax, dparallax, teff, dteff, logz, dlogz, logg, dlogg,
# extprior, extpriorsig


def fit_model(star, nextcomp=2, guess=None, fitpar=[], order=3, niter=3*10**5,
              parallel=0, factr=10**7, mvec=10):
    nstar = star['mag'].shape[0]
    nband = star['mag'].shape[1]
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
        starguess = {}
        starguess['extinction'] = star['extprior'].copy()
        starguess['mu'] = parallax_to_mu(
            numpy.clip(star['parallax'], 0.01, numpy.inf))
        for parname in fitpar:
            starguess[parname] = star[parname]
    else:
        intmagparguess, extcurveparguess, starguess = guess
    guess = wrap_param(intmagparguess, extcurveparguess,
                       starguess, nband, nintmagpar, nstar,
                       fitpar=fitpar)
    def chi2_wrapper(param):
        upar = unwrap_param(param, nband, nintmagpar, nstar, nextcomp, 
                            fitpar=fitpar)
        intmagpar, extcurvepar, starpar = upar
        chi2, grad = fullmodel_chi2(star,
                                    intmagpar, extcurvepar, starpar, 
                                    nband=nband, 
                                    gradient=True, fitpar=fitpar,
                                    order=order)
        return chi2, grad
    if parallel > 0:
        from multiprocessing import Queue, Process
        from multiprocessing.sharedctypes import Array
        import ctypes
        qins = [Queue() for i in range(parallel)]
        qout = Queue()
        ind = numpy.floor(numpy.linspace(0, nstar, parallel+1, endpoint=True))
        ind = ind.astype('i4')
        npar = (nintmagpar+nextcomp)*nband+nstar*(1+nextcomp+len(fitpar))
        grad = Array(ctypes.c_double, npar)
        gradnp = numpy.frombuffer(grad.get_obj(), dtype='f8')
        proclist = [
            Process(target=worker,
                    args=(qins[i], qout, grad, star, nextcomp, fitpar, order,
                          ind[i:i+2], i))
            for i in range(parallel)]
        for p in proclist:
            p.start()
        def chi2_wrapper_parallel(param):
            gradnp[:] = 0.
            for i in range(parallel):
                qins[i].put(param)
            chi2 = 0.
            for i in range(parallel):
                tchi2 = qout.get()
                chi2 += tchi2
            # tgrad gets filled in by the workers; it's shared.
            return chi2, gradnp
        wrapper = chi2_wrapper_parallel
    else:
        wrapper = chi2_wrapper
    from scipy.optimize import fmin_l_bfgs_b, fmin_cg
    res = fmin_l_bfgs_b(wrapper, guess, m=mvec, iprint=10,
                        maxiter=niter, maxfun=niter, factr=factr)
    # cg_chi2_wrapper = lambda x: wrapper(x)[0]
    # cg_grad_wrapper = lambda x: wrapper(x)[1]
    # res = fmin_cg(cg_chi2_wrapper, guess, fprime=cg_grad_wrapper, 
    #               maxiter=niter)
    return res


def worker(qin, qout, grad, star, nextcomp, fitpar, order, ind, pind):
    nband = star['mag'].shape[1]
    nstar = star['mag'].shape[0]
    nextcomp = star['extprior'].shape[1]
    npar = len(grad)
    nperstarpar = 1+nextcomp+len(fitpar)
    nintmagpar = (npar-nextcomp*nband-nstar*nperstarpar)//nband
    try:
        names = star.dtype.names
    except:
        names = star.keys()
    tstar = {name: star[name][ind[0]:ind[1], ...] for name in names}
    
    while True:
        param = qin.get()
        if param is None:
            break
        upar = unwrap_param(param, nband, nintmagpar, nstar, nextcomp, 
                            fitpar=fitpar)
        intmagpar, extcurvepar, starpar = upar
        tstarpar = {name: starpar[name][ind[0]:ind[1], ...] for name in starpar}
        chi2, tgrad = fullmodel_chi2(tstar, intmagpar, extcurvepar, tstarpar,
                                     nband=nband, gradient=True, fitpar=fitpar,
                                     order=order, 
                                     apply_extcurvenormprior=(pind == 0))
        nsharedpar = (nintmagpar + nextcomp)*nband
        tnstar = ind[1]-ind[0]
        ngrad = numpy.frombuffer(grad.get_obj(), dtype='f8')
        with grad.get_lock():
            ngrad[0:nsharedpar] += tgrad[0:nsharedpar]
        for i in range(nperstarpar):
            ngrad[nsharedpar+nstar*i+ind[0]:nsharedpar+nstar*i+ind[1]] = (
                tgrad[nsharedpar+tnstar*i:nsharedpar+tnstar*(i+1)])
        qout.put(chi2)


def numerical_gradients(chi2gradfun, param, dq=1e-5):
    chi2, grad = chi2gradfun(param)
    res = []
    for i in range(len(param)):
        newparam = param + dq*(numpy.arange(len(param)) == i)
        chi20 = chi2gradfun(newparam)[0]
        res.append((chi20-chi2)/dq)
    return numpy.array(res)


def mask(ob):
    badflags1mask = 0
    badflags2mask = 0
    badflags3mask = 0
    badflags1 = [7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                 25, 26, 27, 28, 29, 30]
    badflags2 = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22]
    badflags3 = [8, 24, 10, 26]  # chi2 warn, rotation warn, bad
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


def xmatch_to_good_output(xmatch, nextcomp=0):
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
    dtype = [('ra', 'f8'), ('dec', 'f8'), 
             ('l', 'f8'), ('b', 'f8'),
             ('mag', 'f8', 12), ('dmag', 'f8', 12),
             ('parallax', 'f8'), ('dparallax', 'f8'),
             ('teff', 'f8'), ('dteff', 'f8'),
             ('logz', 'f8'), ('dlogz', 'f8'),
             ('logg', 'f8'), ('dlogg', 'f8')]
    if nextcomp > 0:
        dtype += [('extprior', '%df8' % nextcomp),
                  ('extpriorsig', '%df8' % nextcomp)]
    res = numpy.zeros(len(xmatch), 
                      dtype=dtype)
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
    if nextcomp > 0:
        res['extpriorsig'] = numpy.inf
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
    extprior = numpy.zeros((len(ob), 2), dtype='f8')
    extpriorsig = numpy.zeros((len(ob), 2), dtype='f8')
    extprior[m, 0] = dust.getval(ob['l'][m], ob['b'][m], map='sfd')*avfac
    extpriorsig[m, 0] = 0.1*extprior[m, 0]
    extpriorsig[m, 0] = numpy.sqrt(extpriorsig[m, 0]**2. + (0.01*avfac)**2.)
    extpriorsig[~m, 0] = 30
    extpriorsig[m, 1] = extpriorsig[m, 0]
    extpriorsig[~m, 1] = 3
    # really rough guesses!
    return extprior, extpriorsig


def redclump_cut_bovy14(teff, logz, logg, jk0):
    tref = -382.5*logz+4607
    z = 0.017*10.**logz
    m = logg >= 1.8
    m &= logg <= 0.0018*(teff - tref)+2.5
    # most stars in APOGEE selected from Spitzer IR photometry
    # this changes JK0 relative to Bovy's original cut slightly
    # (extinction correction on J-K depends on W2/Spitzer)
    m &= z > 1.21*(jk0-0.05)**9+0.0011
    m &= z < 2.58*(jk0-0.40)**3+0.0034
    m &= (jk0 > 0.5) & (jk0 < 0.8) & (z < 0.06)
    m &= logg < 0.001*(teff - 4800)+2.75
    return m


def jk0(j, h, k, w2):
    # Zasowski+2013
    return (j - k)-1.5*(0.918*(h-w2-0.05))


def gaiaflag(ob):
    # Gaia DR2 HRD, Babusiaux+2018, performs this cut:
    # AND phot_bp_rp_excess_factor < 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)
    # AND phot_bp_rp_excess_factor > 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)
    # AND visibility_periods_used>8
    # AND astrometric_chi2_al/(astrometric_n_good_obs_al-5)<1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))
    # plus additional cuts on SN in BP, RP, G, and parallax, that we hope the 
    # uncertainties already cover.
    # we duplicate that here.
    bpmrp = ob['phot_bp_mean_mag']-ob['phot_rp_mean_mag']
    bprpexcess = ob['phot_bp_rp_excess_factor']
    m = bprpexcess < 1.3+0.06*bpmrp**2
    m &= bprpexcess > 1.0+0.015*bpmrp**2
    m &= ob['visibility_periods_used'] > 8
    m &= (ob['astrometric_chi2_al']/(ob['astrometric_n_good_obs_al']-5) <
          1.44*numpy.clip(numpy.exp(-0.4*(ob['phot_g_mean_mag']-19.5)), 
                          1, numpy.inf))
    return ~m


# what if life is hard and just doing fmin_l_bfgs_b is not enough?
# probably return to same ~bilinear approach we used before.
# fix the global parameters, solve for the individual stars one-by-one
# fix the individual stellar parameters, solve for the global parameters.

# def fit_model(star, nextcomp=2, guess=None, fitpar=[], order=3, niter=3*10**5,
#               parallel=0, factr=10**7):
#         upar = unwrap_param(param, nband, nintmagpar, nstar, nextcomp, 
#                             fitpar=fitpar)
#         intmagpar, extcurvepar, starpar = upar
#         tstarpar = {name: starpar[name][ind[0]:ind[1], ...] for name in starpar}
#         chi2, tgrad = fullmodel_chi2(tstar, intmagpar, extcurvepar, tstarpar,
#                                      nband=nband, gradient=True, fitpar=fitpar,
#                                      order=order, 
#                                      apply_extcurvenormprior=(pind == 0))
# 
# def fit_stars(star, intmagpar, extcurvepar, starpar, fitpar=[],
#               order=3, damp=3, nextcomp=2):
#     nband = star['mag'].shape[1]
#     for i in range(len(star)):
#         tstarpar = {name: starpar[name][i:i+1] for name in starpar}
#         def chi(param):
#             tstarpar['extinction'] = param[
#             mag = fullmodel(intmagpar, extcurvepar, tstarpar,
#                             nband=nband, gradient=False,
#                             fitpar=fitpar, order=order)
#             return damper((star['mag'][i]-mag)/star['dmag'][i], damp)
#         def grad(param):
            
            
