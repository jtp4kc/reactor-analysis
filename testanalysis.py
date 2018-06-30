'''
Created on Mar 2, 2018

@author: adjun_000
'''

import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.sparse as sparse
import scipy.optimize as optim
import scipy.fftpack as fft
import scipy.interpolate as interp
import matplotlib.pyplot as pyplot

import ReactorAnalysisV6 as ra

def plot_xy(np_array):
    pyplot.plot(np_array[:, 0], np_array[:, 1])

def value_filter(xdata, ydata, minval=None, maxval=None):
    n = np.size(xdata)
    if minval is None:
        minval = np.min(ydata)
    if maxval is None:
        maxval = np.max(ydata)
    inds = []
    for i in xrange(0, n):
        y = ydata[i]
        if minval <= y and y <= maxval:
            inds.append(i)
    grab = (np.array(inds),)
    return xdata[grab], ydata[grab]

def filler_filter(xdata, ydata):
    n = np.size(xdata)
    lx = xdata[0]
    ly = ydata[0]
    ld = 0
    inds = []
    for i in xrange(1, n):
        x = xdata[i]
        y = ydata[i]
        dydx = (y - ly) / (x - lx)
        if dydx != 0 and (np.sign(dydx) != np.sign(ld)):
            ld = dydx
            inds.append(i - 1)
        lx = x
        ly = y
    inds.append(n - 1)
    grab = (np.array(inds),)
    return xdata[grab], ydata[grab]

def bandpass_filter(signal, lowfreq=None, highfreq=None):
    if highfreq is None:
        highfreq = len(signal) / 2 + 1
    if lowfreq is None:
        lowfreq = 1
    w = fft.fft(signal)
    zerofreq = w[0]
    w[:2 * lowfreq - 1] = 0
    w[2 * highfreq - 1:] = 0
    w[0] = zerofreq
    return fft.ifft(w)

def frequency_filter(signal, percmax=0):
    """ Uses FFT, and therefore, assumes that the data is sampled evenly in the
    [time] domain. Otherwise, the frequency and time domains are not correlated.
    @param percmax: percentage of the maximum frequency to filter below
    """
    w = fft.fft(signal)
    s = w ** 2
    mask = s < (s.max() * percmax)
    zerofreq = w[0]
    w[mask] = 0
    w[0] = zerofreq
    return fft.ifft(w)

def baseline_alss(ydata, lamd, p, n_iter=10):
    """ Algorithm from reference:
    | Baseline Correction with Asymmetric Least Squares Smoothing
    | Paul H. C. Eilers Hans F.M. Boelens
    | October 21, 2005
    | AUTHORS' ADDRESSES: Department of Medical Statistics, 
    | Leiden University Medical Centre, 
    | P.O. Box 9604, 2300 RC Leiden, The Netherlands
    | Biosystems Data Analysis Group, 
    | Swammerdam Institute for Life Sciences (SILS), University of Amsterdam, 
    | Nieuwe Achtergracht 166, 1018 WV Amsterdam, The Netherlands.
    | E-MAIL: p.eilers@lumc.nl, hans.boelens@science.uva.nl :
    
    Combines the Whittaker smoothing function (parameter lambda) with
    asymmetric least squares (parameter p, to weight against negative 
    deviations) to fit the baseline of the data
    Try for: 0.001 <= p <= 0.1 (for a signal with positive peaks)
        and: 10^2 <= lambda <= 10^9
    
    S = sum[ w_i * (y_i - z_i)^2 ] + lambda * sum[ (diff^2 z_i)^2 ]
    diff^2 z_i = (z_i - z_i-1) - (z_i-1 - z_i-2) = z_i - 2 z_i-1 + z_i-2
    """
    # Estimate baseline with asymmetric least squares
    m = len(ydata)
    # assemble difference matrix
    D0 = sparse.eye(m)
    D1 = sparse.diags([np.ones(m - 1) * -2], [-1])
    D2 = sparse.diags([np.ones(m - 2) * 1], [-2])
    D = D0 + D1 + D2  # z_i - 2 z_i-1 + z_i-2.
    w = np.ones(m, np.float32)  # iterative weights
    for _ in xrange(n_iter):
        W = sparse.diags([w], [0])  # convert weights to diagonal matrix
        C = W + lamd * D.dot(D.transpose())  # set up system of equations
        z = sparse.linalg.spsolve(C, w * ydata)  # solve system
        w = p * (ydata > z) + (1 - p) * (ydata < z)  # modify weights

    return z  # the baseline

def poly_fit(xdata, ydata, int_order=1, p0=None, xtra=False):
    if p0 is None:
        p0 = [1.0] * (int_order + 1)

    # curve fitting
    def func(x, *p):  # , c, d, e):
        summa = 0
        for i in range(int_order + 1):
            summa += p[i] * x ** i;
        return summa

    params, _ = optim.curve_fit(func, xdata, ydata, p0)
    ret = func(xdata, *params)
    if xtra:
        ret = (ret, func, params)
    return ret

def spline_fit(xdata, ydata):

    def moving_average(series, window=39, sigma=3):
        b = signal.gaussian(window, sigma)
        average = ndimage.filters.convolve1d(series, b / b.sum())
        var = ndimage.filters.convolve1d(np.power(series - average, 2), b / b.sum())
        return average, var

    _, var = moving_average(ydata)
    return interp.UnivariateSpline(xdata, ydata, w=1 / np.sqrt(var))

def linear_interpolate(xvalues, sourcex, sourcey):
    ''' Assumes sourcex, xvalues are in ascending order
    Extrapolates at ends, if sourcex is not inclusive
    '''
    ret = np.zeros_like(xvalues)
    n = np.size(xvalues)
    p = np.size(sourcex)
    s = 0
    lastx = sourcex[0]
    lasty = sourcey[0]
    m = (sourcey[1] - lasty) / (sourcex[1] - lastx)
    for i in xrange(n):
        while s + 2 < p and xvalues[i] > sourcex[s + 1]:
            s += 1
            lastx = sourcex[s]
            lasty = sourcey[s]
            m = (sourcey[s + 1] - lasty) / (sourcex[s + 1] - lastx)
        dx = xvalues[i] - lastx
        ret[i] = lasty + m * dx
    return ret

def smooth_exp(values, alpha=0.5, est=0.001):
    ret = np.zeros_like(values)
    n = np.size(values)
    ret[0] = values[0]
    est_period = int(n * est)
    a = alpha
    if isinstance(alpha, (list, tuple)):
        a = alpha[0]
    for i in xrange(1, est_period):
        ret[i] = a * values[i] + (1 - a) * ret[i - 1]
    if (est_period > 0):
        ret[0] = np.average(ret[:est_period])
    # simple exponential smoothing
    for i in xrange(1, n):
        ret[i] = a * values[i] + (1 - a) * ret[i - 1]
    if isinstance(alpha, (list, tuple)):
        for a in alpha[1:]:
            for i in xrange(1, n):
                ret[i] = a * ret[i] + (1 - a) * ret[i - 1]
    return ret

def smooth_window_avg(values, half_window):
    ret = np.zeros_like(values)
    n = np.size(values)
    for i in xrange(n):
        down = i - half_window
        up = i + half_window
        if down < 0:
            down = 0
        if up >= n:
            up = n - 1
        ret[i] = np.average(values[down:up])
    return ret

def smooth_window_deriv(xdata, ydata, half_window, segment=1):
    n = np.size(ydata)
    npts = n / segment
    xvals = np.zeros(npts, xdata.dtype)
    yvals = np.zeros(npts, ydata.dtype)
    xvals[0] = xdata[0]
    yvals[0] = ydata[0]
    for i in xrange(1, npts):
        down = i * segment - half_window
        h = i * segment
        up = i * segment + half_window
        if down < 0:
            down = 0
        if h >= n:
            h = n - 1
        if up >= n:
            up = n - 1
        windx = xdata[down:up]
        windy = ydata[down:up]
        dydx = np.sum(windy[1:] - windy[:-1]) / np.sum(windx[1:] - windx[:-1])
        xvals[i] = xdata[h]
        yvals[i] = yvals[i - 1] + dydx * (xvals[i] - xvals[i - 1])
    return linear_interpolate(xdata, xvals, yvals)

def smooth_piecewise_line(xdata, ydata, window):
    n = np.size(ydata)
    r = n / window
    xvals = np.zeros(2 * r + 3, xdata.dtype) * np.NaN
    yvals = np.zeros(2 * r + 3, ydata.dtype) * np.NaN
    for w in xrange(r + 1):
        down = w * window
        i = int((w + 0.5) * window)
        up = (w + 1) * window
        if down >= n:
            down = n - 1
        if i >= n:
            i = n - 1
        if up >= n:
            up = n - 1
        if down == up:
            xvals[2 * w + 0] = xdata[down]
            yvals[2 * w + 0] = ydata[down]
        else:
            rngx = xdata[down:up]
            rngy = ydata[down:up]
            _, func, p = poly_fit(rngx, rngy, int_order=1, xtra=True)
            xvals[2 * w + 0] = xdata[down]
            xvals[2 * w + 1] = xdata[i]
            if w > 0:
                yvals[2 * w + 0] += func(xdata[down], *p)
                yvals[2 * w + 0] /= 2
            else:
                yvals[2 * w + 0] = func(xdata[down], *p)
            yvals[2 * w + 1] = func(xdata[i], *p)
            yvals[2 * w + 2] = func(xdata[up], *p)
    return linear_interpolate(xdata, xvals, yvals)

def baseline_filter(xdata, ydata):
    xdataorig = xdata
    ydataorig = ydata
    mask = np.where(~np.isnan(ydata))
    xdata = xdata[mask]
    ydata = ydata[mask]

    n = np.size(xdata)
    bl3 = smooth_piecewise_line(xdata, ydata, n / 250)

    slopeup = 600  # counts per min
    slopedown = -1000
    derivs = np.zeros_like(bl3)
    derivs[:-1] = bl3[1:] - bl3[:-1]
    derivs[:-1] /= (xdata[1:] - xdata[:-1])
    criteria = np.logical_and(derivs < slopeup, derivs > slopedown)
    criteria[0] = True
    mask = np.where(criteria)
    mask2 = np.where(~criteria)
    derivs[mask] = np.NaN
    bl3_ = np.zeros_like(bl3)
    bl3_[:] = bl3
    bl3_[mask2] = np.NaN
    df = ydata - bl3_
    avgsse = np.nansum(df ** 2) / n
    erval = np.sqrt(avgsse)

    project = np.zeros_like(bl3) * np.NaN
    xlast = 0
    lastval = np.NaN
    deriv = 300
    for i in xrange(n):
        if np.isnan(lastval):
            xlast = xdata[i]
            lastval = bl3_[i]
        elif not np.isnan(bl3_[i]):
            est = deriv * (xdata[i] - xlast) + lastval
            if ((bl3_[i] - lastval) > erval * 3.00
                    and (bl3_[i] - est) > erval * 3.00):
                bl3_[i] = np.NaN
                project[i] = est
            else:
                deriv = 300  # (bl3_[i] - lastval) / (xdata[i] - xlast)
                xlast = xdata[i]
                lastval = bl3_[i]

    floor = bl3_ - 2.0 * erval
    mask = np.where(~np.isnan(floor))
    xvals = xdata[mask]
    floor = floor[mask]
    # floor = baseline_alss(floor, 500, 0.01)
    floor = linear_interpolate(xdataorig, xvals, floor)

    fltr = np.array(ydataorig)
    fltr[np.where(np.logical_or(fltr < floor,
                                fltr > (floor + 6 * erval)))] = np.NaN
    return fltr

def derivative(x, y):
    ''' Calculates the instantaneous derivative of a signal using two-point
    differences. Inputs are 1-D numpy arrays of the same shape.
    '''
    if x.shape != y.shape or len(x.shape) != 1:
        raise ValueError("Input arrays must be 1-D and have the same shape. (x:" +
                         str(x.shape) + ", y:" + str(y.shape) + ")")
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    deriv = dy / dx
    deriv = np.concatenate((deriv, [0]))
    return deriv

class Line():

    def __init__(self):
        self.m = 0
        self.b = 0
        self.x1 = -1
        self.x2 = -1
        self.xvals = None
        self.yvals = None
        self.ydata = None

def line_walk(x, y, window=10, minwidth=50, cutoff=500):

    def linear(x, *p):
        return p[1] * x + p[0]

    lines = []
    last = 0
    prevfit = None
    for i in xrange(0, x.shape[0], window):
        if i < last + minwidth:
            continue
        buffx = x[last:i]
        buffy = y[last:i]
        guessm = (buffy[-1] - buffy[0]) / (buffx[-1] - buffx[0])
        guessb = buffy[0] - guessm * buffx[0]
        if np.isnan(guessm) or np.isnan(guessb):
            guessm = guessb = 1
        popt, _ = optim.curve_fit(linear, buffx, buffy, [guessb, guessm])
        # perr = np.sqrt(np.diag(pcov))
        esty = linear(buffx, *popt)
        sse = np.sqrt(np.sum((buffy - esty) ** 2))
        # stdev = np.std(buffy - esty)
        if sse > cutoff:
            if prevfit is not None:
                # append the parameters as a new line
                ind = i - window
                lin = Line()
                lin.b = prevfit[0]
                lin.m = prevfit[1]
                lin.x1 = x[last]
                lin.x2 = x[ind]
                lin.xvals = x[last:ind]
                lin.yvals = linear(lin.xvals, *prevfit)
                lin.ydata = y[last:ind]
                lines.append(lin)
                prevfit = None
                last = ind
            else:
                # the previous segment was not linear, advance to new segment
                last += window
        else:
            # the current segment is linear, accept as a segment, test next window
            prevfit = popt
    return lines

class SmoothingCurve():

    def __init__(self):
        self.y0 = 0
        self.d0 = 0
        self.accel = None

    def generate(self, x):

        def midpoint(a1, a2, val, lastind):
            bef = -1
            aft = -1
            lastind -= 1
            if lastind < 0:
                lastind = 0
            for ind in xrange(lastind, len(a1)):
                if a1[ind] <= val:
                    bef = ind
                if a1[ind] >= val:
                    aft = ind
                    lastind = ind
                    break
            xrng = a1[aft] - a1[bef]
            if xrng == 0:
                mid = a2[bef]
            else:
                perc = (val - a1[bef]) / xrng
                mid = (a2[aft] - a2[bef]) * perc + a2[bef]
            return mid, lastind

        y = np.zeros_like(x, np.float64)
        y[0] = self.y0
        d = self.d0
        last = 0
        for i in xrange(1, y.shape[0]):
            # use midpoint to interpolate the accel values, fill in data points
            a, last = midpoint(self.accel[0, :], self.accel[1, :], x[i], last)
            d += a * (x[i] - x[i - 1])
            y[i] = d * (x[i] - x[i - 1]) + y[i - 1]
        self.x = x
        self.y = y
        return y

    def test(self):
        self.accel = np.array([np.arange(21) * 5, np.random.random(21) - 0.5])
        self.d0 = 0
        self.y0 = 0

        return self.generate(np.arange(1000) / 10.0)

def testsmooth(npts=20):
    curve = SmoothingCurve()
    curve.accel = np.zeros((2, npts), np.float64)
    spacing = np.linspace(0, dx[-1], npts)
    for i in range(npts):
        curve.accel[0, i] = spacing[i]

    def unpack_curve(x, *p):
        curve.y0 = p[0]
        curve.d0 = p[1]
        for i in range(npts):
            curve.accel[1, i] = p[i + 2]
        return curve.generate(x)

    p0 = np.array([-5000, 0] + [0] * npts)
    param, _ = optim.curve_fit(unpack_curve, dx, dy, p0)
    unpack_curve(dx, *param)
    return curve

def custom_smooth(x, y, y0win=50, d0win=100, ratio=20):
    n = x.shape[0]
    y0 = np.average(y[:y0win])
    d0 = np.average((y[1:d0win] - y[:d0win - 1]) / (x[1:d0win] - x[:d0win - 1]))
    curve = SmoothingCurve()
    curve.y0 = y0; curve.d0 = d0
    curve.accel = np.zeros((2, n), np.float64)
    x0 = x[0]
    a0 = 0
    val = y0
    for i in xrange(1, n):
        x1 = x[i]
        y1 = y[i]
        d1 = (y1 - y0) / (x1 - x0)
        a1 = (d1 - d0) / (x1 - x0)
        diff = y0 - val
        factor = np.exp(-diff / ratio)
        a1 = a1 * factor
        curve.accel[0, i - 1] = x1
        curve.accel[1, i - 1] = a1
        val += a1 * (x1 - x0) ** 2
        x0 = x1; y0 = y1; d0 = d1

#     replace = np.zeros((2, n / ratio), np.float64)
#     for i in xrange(n / ratio):
#         ind = i * ratio
#         replace[0, i] = curve.accel[0, ind + ratio - 1]
#         replace[1, i] = np.average(curve.accel[1, ind:ind + ratio])
#    curve.accel = replace

    return curve

def integrate_signal(x, y):
    smooth = smooth_window_deriv(x, y, 50)
    deriv = derivative(x, smooth)
    fltx, flty = filler_filter(x, deriv)

    """ Create an algorithm that coasts along the signal, plotting a smooth line to 
    fit the data, and segments it into portions that can each be optimized to 
    achieve a better fit. When the signal deviates too much from the line, stiffen
    the curve so that it moves more slowly (lower derivative). This will help create
    a baseline underneath the peaks, and help with the giant tailing peaks problem
    
    Do peak fitting in order to de-convolute peaks that co-ellute, that is, to draw 
    a line under a peak that comes out on the tail of an earlier peak. See the work
    with second order process responses for an example of peak shape 
    """

def shimazdu_integrate(x, y, width=8, slope=1000, drift=800, tdbl=1000,
                       minarea=10000):
    NORM = "Normal"; PEAK = "Peak"
    peaks = []
    base = y[0]
    peakstart = x[0]
    integ = 0
    state = NORM
    for i in xrange(1, x.shape[0]):
        baseold = base
        xold = x[i - 1]
        xval = x[i]
        yold = y[i - 1]
        yval = y[i]
        m = (yval - yold) / ((xval - xold) * 60)
        base += (xval - xold) / 60.0 * drift
        if yval < base:
            base = yval
            peakstart = xval
            integ = 0
        if m >= slope:
            trypeak = True
        if trypeak:
            pass

testdata = ra.import_shimadzu_data("batch_data043.txt")
data = np.array(testdata.chromatogram)
dx = data[:, 0]
dy = data[:, 1]
fltr = filler_filter(dx, dy)
fx = fltr[0]
fy = fltr[1]

if __name__ == '__main__':
    pass

