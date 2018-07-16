'''
Created on Mar 2, 2018

@author: adjun_000
'''

import os
import math
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.sparse as sparse
import scipy.optimize as optim
import scipy.fftpack as fft
import scipy.interpolate as interp
import matplotlib.pyplot as pyplot

FILE_SAVE = "_integrate_v01.sav"
FILE_TEST = "../Sandbox/batch_data043.txt"

class obj(object):
    pass

def import_shimadzu_data(filename, delimiter="\t", headerlines=0,
                         quotechar='"',
                         smplflag="[Sample Information]", smpl_lead=1,
                         peakflag="[Peak Table", peak_lead=2,
                         cmpdflag="[Compound Results", leading=2,
                         crmgflag="[Chromatogram", crmg_lead=5):
    ret = obj()
    ret.peaks = list()
    ret.cmpds = list()
    ret.chromatogram = list()
    ret.time = ""
    with open(filename) as target:
        skip = 0
        mark = -1
        mark2 = -1
        mark3 = -1
        mark4 = -1
        for line in target:
            skip += 1
            if skip <= headerlines:
                continue
            if line.startswith(cmpdflag):
                mark = 0
            if line.startswith(smplflag):
                mark2 = 0
            if line.startswith(peakflag):
                mark3 = 0
            if line.startswith(crmgflag):
                mark4 = 0
            if mark >= 0:
                mark += 1
                if "" == line.strip():
                    mark = -1
            if mark2 >= 0:
                mark2 += 1
                if "" == line.strip():
                    mark2 = -1
            if mark3 >= 0:
                mark3 += 1
                if "" == line.strip():
                    mark3 = -1
            if mark4 >= 0:
                mark4 += 1
                if "" == line.strip():
                    mark4 = -1
            if mark2 > smpl_lead + 1:
                splt = line.split(delimiter)
                cnt = 0
                for s in splt:
                    cnt += 1
                    st = s.strip()
                    if (cnt == 2):
                        ret.time = st
                mark2 = -1
            if mark3 > peak_lead + 1:
                splt = line.split(delimiter)
                if len(splt) > 4:
                    ret.peaks.append(
                            (int(splt[0].strip()),  # peak no
                            (float(splt[1].strip())),  # ret time
                            (float(splt[4].strip()))))  # peak area
            if mark4 > crmg_lead + 1:
                splt = line.split(delimiter)
                try:
                    ret.chromatogram.append((float(splt[0]), int(splt[1])))
                except ValueError as ve:
                    print("Error reading chromatogram at line " + str(skip))
                    raise ve
            if mark > leading + 1:
                splt = line.split(delimiter)
                row = list()
                cnt = 0
                for s in splt:
                    cnt += 1
                    st = s.strip()
                    if (cnt >= 2 and cnt <= 4):
                        row.append(st)
                ret.cmpds.append(row)

    return ret

def plot_xy(np_array):
    pyplot.plot(np_array[:, 0], np_array[:, 1])

def trapz(X, Y=None):
    if (Y is None):
        Y = X
        X = np.arange(len(Y))
    vals = (Y[1:] + Y[:-1]) * (X[1:] - X[:-1]) / 2
    return vals.sum()

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

class Spline():

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

def moving_average(data, span=5):
    n = len(data)
    D = sparse.eye(n)
    half = int(math.floor(span / 2))
    for i in range(half + 1 - span, half + 1):
        if i != 0:
            D += sparse.eye(n, k=i)

    val = D.sum(0).A[0]  # converts matrix return to a 1-D array
    summa = D.dot(data)
    return summa / val

def custom_smooth(x, y, y0win=50, d0win=100, ratio=20, pan=5, y0override=None, d0override=None):
    n = x.shape[0]
    y0 = np.average(y[:y0win])
    d0 = np.average((y[1:d0win] - y[:d0win - 1]) / (x[1:d0win] - x[:d0win - 1]))
    if y0override is not None:
        y0 = y0override
    if d0override is not None:
        d0 = d0override
    curve = Spline()
    curve.y0 = y0; curve.d0 = d0
    curve.accel = np.zeros((2, n), np.float64)
    x0 = x[pan]
    a0 = 0
    val = y0
    for i in xrange(pan + 1, n - 1):
        x1 = x[i]
        y1 = y[i - pan:i + 1]
        d1 = np.average((y1 - y0) / (x1 - x0))
        a1 = (d1 - d0) / (x1 - x0)
        diff = y0 - val
        if ratio == 0:
            factor = 1.0
        else:
            factor = np.exp(-diff / ratio)
        a1 = a1 * factor  # + (1 - factor) * a0
        curve.accel[0, i - 1] = x1
        curve.accel[1, i - 1] = a1
        val += (a1 * (x1 - x0) + d1) * (x1 - x0)
        x0 = x1; y0 = y1[-1]; d0 = d1; a0 = a1

#     replace = np.zeros((2, n / ratio), np.float64)
#     for i in xrange(n / ratio):
#         ind = i * ratio
#         replace[0, i] = curve.accel[0, ind + ratio - 1]
#         replace[1, i] = np.average(curve.accel[1, ind:ind + ratio])
#    curve.accel = replace

    return curve

def custom_smooth2(x, y, window=101, slope=1000, tol=1000, npts=25):
    n = len(x)
    ma = moving_average(y, window)
    da = derivative(x, ma)
    ind = []
    lock = 0
    peak = False
    count = 0
    start = None
    updown = None
    last = None
    for i in range(1, n):
        if da[i] > 0 and last == "DOWN":
            updown = "UP"
        elif da[i] < 0 and last == "UP":
            updown = "DOWN"
        else:
            updown = None
        if da[i] > 0:
            last = "UP"
        elif da[i] < 0:
            last = "DOWN"
        if not peak and da[i] > slope:
            # report first x where this occurs
            peak = True
            start = i
            lock = i - 1
            count = 20
        if count > 0:
            count -= 1
        elif peak and updown == "UP" and np.abs(ma[i] - ma[lock]) < tol:
            peak = False
            if start is not None:
                ind.append((start, i))
                start = None
    buildx = []
    buildy = []
    last = 0
    for i in ind:
        buildx.append(x[last:i[0]])
        buildy.append(ma[last:i[0]])
        last = i[1]
    xs = np.concatenate(buildx)
    ys = np.concatenate(buildy)

    curve = Spline()
    curve.accel = np.zeros((2, npts), np.float64)
    spacing = np.linspace(0, xs[-1], npts)
    for i in range(npts):
        curve.accel[0, i] = spacing[i]

    def unpack_curve(mx, *p):
        curve.y0 = p[0]
        curve.d0 = p[1]
        for i in range(npts):
            curve.accel[1, i] = p[i + 2]
        return curve.generate(mx)

    p0 = np.array([ma[0], 0] + [0] * npts)
    param, _ = optim.curve_fit(unpack_curve, xs, ys, p0)
    vals = unpack_curve(xs, *param)
    return linear_interpolate(x, xs, vals)

class Peake():

    def __init__(self):
        self.time = None
        self.orig = None
        self.base = None
        self.data = None
        self.retention = 0
        self.leading = 0
        self.tailing = 0
        self.area = 0
        self.height = 0
        self.width = 0
        self.negative = False
        self.clips = False

    def calculate(self, leading=None, tailing=None):
        self.data = self.orig - self.base
        self.leading = self.time[0]
        if leading is not None:
            self.leading = leading
        self.tailing = self.time[-1]
        if tailing is not None:
            self.tailing = tailing
        self.negative = np.sign(self.data).sum() < 0
        neg = 1
        self.height = np.max(self.data)
        self.width = self.tailing - self.leading
        self.retention = self.time[np.argmax(self.data)]
        if self.negative:
            neg = -1
            self.height = -np.min(self.data)
            self.retention = self.time[np.argmin(self.data)]
        self.area = 60 * trapz(self.time, self.data) * neg
        self.clips = False
        startpeak = False
        endpeak = False
        for i in xrange(len(self.time)):
            t = self.time[i]
            d = neg * self.data[i]
            if t < self.leading:
                continue
            if not startpeak and d > 0:
                startpeak = True
            if startpeak and not endpeak and d <= 0:
                endpeak = True
            if endpeak and d > 0:
                self.clips = True
                break
            if t > self.tailing:
                break

    def try_split(self):
        self.fit_peak()
        return None

    def fit_peak(self):
        time = self.time - self.time[0]
        neg = 1
        if self.negative:
            neg = -1
        data = neg * self.data
#         curve = self.perform_fitting(time, data)
        curve = moving_average(data, 101)
        le = local_extrema(time, curve)
        segs = []
        width = 8.0 / 60
        start = 0
        last = 0
        for i in xrange(1, len(time)):
            if le[i] < 0 and time[i] - last > width:
                segs.append((self.time[start:i], self.data[start:i]))
                start = i
                last = time[i]
        fill = trapz(time, curve) * 60
        pyplot.plot(self.time, neg * curve, "r")
        for (t, d) in segs:
            pyplot.plot(t, d)
        print "Area:", self.area, "Curve:", fill

    def heaviside(self, x):
        y = np.ones_like(x)
        y[x < 0] = 0
        return y

    def fitting_function(self, t, p):
        """ Second order response, kind of
        @see: https://www.springer.com/cda/content/document/cda_downloaddocument/
              9781441910264-c1.pdf?SGWID=0-0-45-1202938-p173970268
        @param: t - time values
        @param: p - fitting parameters
        """
        d1 = np.abs(p[0])
        a = np.abs(p[1])
        b = np.abs(p[2])
        c = np.abs(p[3])
        t1 = t - d1
        f = a * (np.exp(-b * t1) * np.sinh(c * t1) * self.heaviside(t1))
        return f

    def fitting_function2(self, t, p, split=0):
        """ 
        @param: t - time values
        @param: p - fitting parameters
        """
        a = np.abs(p[0])
        b = np.abs(p[1])
        c = np.abs(p[2])
        t1 = t - split
        f1 = a * (np.exp(-b * t1 ** 2))
        f2 = a * (np.exp(-c * t1 ** 2))
        f1[t1 > 0] = 0
        f2[t1 <= 0] = 0
        return f1 + f2

    def fitting_function3(self, t, p, split=0):
        """ 
        @param: t - time values
        @param: p - fitting parameters
        """
        d1 = p[0]
        d2 = np.abs(p[1])
        a = np.abs(p[2])
        b = np.abs(p[3])
        c = np.abs(p[4])
        t1 = t - d1
        t2 = t
        f = a * (np.exp(-b * t1 ** 2 - d2 * np.log(c * t2) ** 2))
        f[t2 <= 0] = 0
        return f

    def perform_fitting(self):
        t = self.time - self.time[0]
        y = self.data[:]

        tmax = t[np.argmax(y)]

        def minfun_(p):
            return self.fitting_function2(t, p, split=tmax) - y

        # results = optim.leastsq(minfun_, np.array([0.1, self.area, 10, 0.01]))
        # results = optim.leastsq(minfun_, np.array([0.1, 0.5, 0, self.area, 10, 1, 1, 0.02]))
        results = optim.leastsq(minfun_, np.array([self.area, 1, 1]))  # f2
        # results = optim.leastsq(minfun_, np.array([1, 0, self.area, 1, 1])) # f3
        results = optim.leastsq(minfun_, results[0])
        # print results
        f = self.fitting_function2(t, results[0], split=tmax)

        pyplot.plot(self.time, f + self.base)

        y = y - f
        tmax = t[np.argmax(y)]
        results = optim.leastsq(minfun_, np.array([y.max(), 1, 1]))
        results = optim.leastsq(minfun_, results[0])
        g = self.fitting_function2(t, results[0], split=tmax)

        pyplot.plot(self.time, g + self.base)

        y = y - g
        tmax = t[np.argmax(y)]
        results = optim.leastsq(minfun_, np.array([y.max(), 1, 1]))
        results = optim.leastsq(minfun_, results[0])
        h = self.fitting_function2(t, results[0], split=tmax)

        pyplot.plot(self.time, h + self.base)

        return f + g + h

def integrate_signal(x, y, reltol=0.05):
    smooth = custom_smooth2(x, y)
    signal = y - smooth

    negsig = signal * -1.0
    negsig[negsig < 0] = 0
    n = len(x)

    peaks = []
    traps = (negsig[1:] + negsig[:-1]) * (x[1:] - x[:-1]) / 2
    startind = -1
    for i in xrange(1, n - 1):
        if negsig[i] > 0:
            if startind < 0:
                startind = i
        else:
            if startind >= 0:
                area = 60 * traps[startind:i].sum()
                if area > 0:
                    peak = Peake()
                    peak.negative = True
                    peak.data = -negsig[startind:i + 1]
                    peak.time = x[startind:i + 1]
                    peak.orig = y[startind:i + 1]
                    peak.base = smooth[startind:i + 1]
                    peak.area = area
                    peak.height = -peak.data.min()
                    peaks.append(peak)
            startind = -1

    modsig = signal * 1.0
    modsig[modsig < 0] = 0

    traps = (modsig[1:] + modsig[:-1]) * (x[1:] - x[:-1]) / 2
    startind = -1
    for i in xrange(1, n - 1):
        if modsig[i] > 0:
            if startind < 0:
                startind = i
        else:
            if startind >= 0:
                area = 60 * traps[startind:i].sum()
                if area > 0:
                    peak = Peake()
                    peak.data = modsig[startind:i + 1]
                    peak.time = x[startind:i + 1]
                    peak.orig = y[startind:i + 1]
                    peak.base = smooth[startind:i + 1]
                    peak.area = area
                    peak.height = peak.data.max()
                    peaks.append(peak)
            startind = -1

    height = np.array([p.height for p in peaks])
    areas = np.array([p.area for p in peaks])
    msd = areas.std()
    accepted = []
    last = msd
    for _ in xrange(len(peaks)):  # maximum of npeaks loops
        ind = np.nanargmax(areas)
        big = areas[ind]
        areas[ind] = np.NaN
        sd = np.nanstd(areas)
        if np.abs(sd - last) < reltol * last:
            areas[ind] = big
            break
        else:
            accepted.append(ind)
        last = sd

    avg = np.nanmean(areas)
    print "Noise in peaks:", avg

    keep = []
    sumheight = height.sum()
    for ind in accepted:
        accept = peaks[ind]
        keep.append(accept)
        sumheight -= accept.height
        add = accept.try_split()
        if add is not None:
            for a in add:
                keep.append(a)
    avh = sumheight / (len(peaks) - len(accepted))
    print "Noise in signal:", avh
    return keep

""" Create an algorithm that coasts along the signal, plotting a smooth line to 
fit the data, and segments it into portions that can each be optimized to 
achieve a better fit. When the signal deviates too much from the line, stiffen
the curve so that it moves more slowly (lower derivative). This will help create
a baseline underneath the peaks, and help with the giant tailing peaks problem

Do peak fitting in order to de-convolute peaks that co-ellute, that is, to draw 
a line under a peak that comes out on the tail of an earlier peak. See the work
with second order process responses for an example of peak shape 
"""

def local_extrema(x, y):
    da = derivative(x, y)
    le = np.zeros_like(y)
    last = da[0]
    for i in xrange(1, len(x)):
        if da[i] != 0:
            if last < 0 and da[i] > 0:
                le[i] = -1
            if last > 0 and da[i] < 0:
                le[i] = 1
            last = da[i]
    if da[0] < 0:
        le[0] = 1
    if da[0] > 0:
        le[0] = -1
    if da[-2] < 0:
        le[-1] = -1
    if da[-2] > 0:
        le[-1] = 1
    return le

def estimate_baseline(x, y, window=101, slope=1000, tol=1000, npts=25):
    n = len(x)
    ma = moving_average(y, window)
    da = derivative(x, ma)
    ind = []
    lock = 0
    peak = False
    count = 0
    start = None
    updown = None
    last = None
    for i in range(1, n):
        if da[i] > 0 and last == "DOWN":
            updown = "UP"
        elif da[i] < 0 and last == "UP":
            updown = "DOWN"
        else:
            updown = None
        if da[i] > 0:
            last = "UP"
        elif da[i] < 0:
            last = "DOWN"
        if not peak and da[i] > slope:
            # report first x where this occurs
            peak = True
            start = i
            lock = i - 1
            count = 20
        if count > 0:
            count -= 1
        elif peak and updown == "UP" and np.abs(ma[i] - ma[lock]) < tol:
            peak = False
            if start is not None:
                ind.append((start, i))
                start = None
    buildx = []
    buildy = []
    last = 0
    nully = np.zeros_like(y) * np.NaN
    for i in ind:
        buildx.append(x[last:i[0]])
        buildy.append(ma[last:i[0]])
        nully[last:i[0]] = ma[last:i[0]]
        last = i[1]
    xs = np.concatenate(buildx)
    ys = np.concatenate(buildy)
    mask = np.isnan(nully)

    curve = Spline()
    curve.accel = np.zeros((2, npts), np.float64)
    spacing = np.linspace(0, x[-1], npts)
    for i in range(npts):
        curve.accel[0, i] = spacing[i]

    def unpack_curve(p):
        curve.y0 = p[0]
        curve.d0 = p[1]
        for i in range(npts):
            curve.accel[1, i] = p[i + 2]
        ret = y - curve.generate(x)
        ret[mask] = 0
        return ret

    p0 = np.array([ma[0], 0] + [0] * npts)
    param = optim.leastsq(unpack_curve, p0)
    unpack_curve(param[0])
    return curve  # , xs

def reject_peaks(x, y, baseline, signal, reltol=0.04):
    negsig = signal * -1.0
    negsig[negsig < 0] = 0
    n = len(x)

    peaks = []
    startind = -1
    for i in xrange(1, n - 1):
        if negsig[i] > 0:
            if startind < 0:
                startind = i
        else:
            if startind >= 0:
                peak = Peake()
                peak.data = -negsig[startind:i + 1]
                peak.time = x[startind:i + 1]
                peak.orig = y[startind:i + 1]
                peak.base = baseline[startind:i + 1]
                peak.calculate()
                peaks.append(peak)
            startind = -1

    modsig = signal * 1.0
    modsig[modsig < 0] = 0

    startind = -1
    for i in xrange(1, n - 1):
        if modsig[i] > 0:
            if startind < 0:
                startind = i
        else:
            if startind >= 0:
                peak = Peake()
                peak.data = modsig[startind:i + 1]
                peak.time = x[startind:i + 1]
                peak.orig = y[startind:i + 1]
                peak.base = baseline[startind:i + 1]
                peak.calculate()
                peaks.append(peak)
            startind = -1

    height = np.array([p.height for p in peaks])
    width = np.array([p.width for p in peaks])
    areas = np.array([p.area for p in peaks])
    msd = areas.std()
    accepted = []
    last = msd
    for _ in xrange(len(peaks)):  # maximum of npeaks loops
        ind = np.nanargmax(areas)
        big = areas[ind]
        areas[ind] = np.NaN
        sd = np.nanstd(areas)
        if np.abs(sd - last) < reltol * last:
            areas[ind] = big
            break
        else:
            accepted.append(ind)
        last = sd

    avg = np.nanmean(areas)

    keep = []
    sumheight = height.sum()
    sumwidth = width.sum()
    for ind in accepted:
        accept = peaks[ind]
        keep.append(accept)
        sumheight -= accept.height
        sumwidth -= accept.width
    avh = sumheight / (len(peaks) - len(accepted))
    avw = sumwidth / (len(peaks) - len(accepted))

    return keep, avg, avh, avw

def resolve_fit(t, p, split=0):
    """ Asymmetric Gaussian fitting
    @param: t - time values
    @param: p - fitting parameters
    """
    a = np.abs(p[0])
    b = np.abs(p[1])
    c = np.abs(p[2])
    t1 = t - split
    f1 = a * (np.exp(-b * t1 ** 2))
    f2 = a * (np.exp(-c * t1 ** 2))
    f1[t1 > 0] = 0
    f2[t1 <= 0] = 0
    return f1 + f2

def resolve_tail(t, p):
    """ Exponential decay
    @param: t - time values
    @param: p - fitting parameters
    """
    a = np.abs(p[0])
    b = np.abs(p[1])
    c = p[2]
    t1 = t - c
    return a * np.exp(-b * t1) * t1

def resolve_fitting(t, y, tmax, ymax, exp=False):

    def minfit(p):
        return resolve_fit(t, p, split=tmax) - y

    def mintail(p):
        return resolve_tail(t, p) - y

    p0 = [ymax, 1, 1]
    fitfun = minfit
    if exp:
        p0 = [ymax, 1, t[0]]
        fitfun = mintail

    results = optim.leastsq(fitfun, np.array(p0))
    results = optim.leastsq(fitfun, results[0])
    return results[0]

def resolve_peaks(peaks, area_noise, sig_noise, wid_noise):
    keep = []
    # print "Area cutoff:", area_noise
    # print "Height cutoff:", sig_noise
    # print "Width cutoff:", wid_noise
    for p in peaks:
        time = p.time - p.time[0]
        orig = np.array(p.orig)
        base = np.array(p.base)
        neg = 1
        if p.negative:
            neg = -1

        temp = []
        n = len(time)
        start = 0
        while start < n:
            data = (orig - base) * neg
            le = local_extrema(time, data)
            up = start
            dn = start
            peakstart = False
            peakend = False
            for i in xrange(start + 1, n):
                if le[i] > 0:
                    up = i
                if le[i] < 0:
                    dn = i
                if not peakstart and le[i] > 0:
                    wid = time[i] - time[dn]
                    if wid > wid_noise / 2:
                        peakstart = True
                if peakstart and le[i] < 0:
                    wid = time[i] - time[up]
                    if wid > wid_noise / 2:
                        peakend = True
                if peakend and le[i] > 0:
                    wid = time[i] - time[dn]
                    if wid > wid_noise / 2:
                        imax = np.argmax(data[start:dn]) + start
                        tmax = time[imax]
                        ymax = data[imax]
                        iend = np.argmin(data[imax:dn]) + imax
                        fit1 = False
                        if fit1:
                            param = resolve_fitting(time[start:iend],
                                    data[start:iend], tmax, ymax)
                            fit = resolve_fit(time, param, split=tmax)
                            fit[:iend] = data[:iend]
                            fit[fit > data] = data[fit > data]
                        else:
                            fall = (ymax - data[iend]) / 2
                            ifit = imax
                            for j in xrange(imax, iend):
                                if ymax - fall > data[j]:
                                    ifit = j
                                    break
                            param = resolve_fitting(time[ifit:iend],
                                    data[ifit:iend], tmax, ymax, exp=True)
                            fit = resolve_tail(time, param)
                            fit[:iend] = data[:iend]
                            fit[fit > data] = data[fit > data]
                        fit *= neg
                        newp = Peake()
                        newp.time = np.array(p.time)
                        newp.orig = np.array(p.orig)
                        newp.base = np.array(base)
                        base = base + fit
                        newp.orig[iend:] = base[iend:]
                        newp.calculate(p.time[start], p.time[iend])
                        temp.insert(0, newp)

                        # pyplot.plot(p.time[iend:], base[iend:])
                        start = iend
                        break
            if i >= n - 1:  # end
                newp = Peake()
                newp.time = np.array(p.time)
                newp.orig = np.array(p.orig)
                newp.base = np.array(base)
                newp.calculate(p.time[start])
                temp.insert(0, newp)
                start = n

        n = len(temp)
        temp2 = []
        for i in xrange(n):  # collapse small peaks
            t = temp[i]
            if t.area > area_noise and t.height > sig_noise:
                temp2.insert(0, t)
                # print "Keeping", t.retention, "A:", t.area, "H:", t.height
            elif i >= n - 1:  # end of list
                if i > 0:
                    o = temp[i - 1]
                    o.base = t.base
                    o.calculate(t.leading, o.tailing)
                    # print "MergingR", t.retention, "A:", t.area, "H:", t.height
                else:
                    temp2.insert(0, t)
                    # print "Keeping", t.retention, "A:", t.area, "H:", t.height
            else:  # add this peak to the next one back
                o = temp[i + 1]
                o.orig = t.orig
                o.calculate(o.leading, t.tailing)
                # print "MergingL", t.retention, "A:", t.area, "H:", t.height
        for t in temp2:  # sort by retention time
            ind = 0
            for i in xrange(len(keep)):
                if t.retention > keep[i].retention:
                    ind = i + 1
            keep.insert(ind, t)
    return keep

# TODO List:
# Try fitting the baseline with the missing values replaced by 0's in the residual,
#     so that linear filling is not needed.
# Move this code to the other file.
# Make it work for both with chromatogram or without (use the existing peak list)
# Plot the baseline and peaks for each file in one document, and record a zoom in
#     of each peak in another, that shows better detail of the as-fit and adjusted
#     signals after integration is complete
# Test this code against a whole reaction run, compare with Shimadzu integration
# Parse the peak list across all files to determine where components are, and filter
#     the minor peaks by which major peak they belong to (EtOH)
# Connect different runs together, to establish the non-reaction datapoints, so that
#     a mass balance can be automatically calculated
# Re-analyze everything! Will need to go through and update background conversions.

def process_datafile(filename=FILE_TEST, ifprint=False):
    importfile = import_shimadzu_data(filename)
    if len(importfile.chromatogram) > 0:
        data = np.array(importfile.chromatogram)
        x = data[:, 0]
        y = data[:, 1]

        fname = filename
        if filename.endswith(".txt"):
            fname = filename.rsplit(".txt", 1)[0]
        sf = Savefile_v01(fname)
        if os.path.exists(sf.filename) and os.path.isfile(sf.filename):
            sf.read()

        if sf.baseline is not None:
            curve = sf.baseline
            xs = curve.xs
        else:
            curve = estimate_baseline(x, y)
            curve.xs = x
            sf.baseline = curve
            sf.save()

        baseline = curve.generate(x)
        # baseline = linear_interpolate(x, xs, vals)
        signal = y - baseline

        # pyplot.plot(x, y, x, baseline)

        if sf.integrate is not None:
            peaks = sf.integrate
            (area, sig, width) = sf.stats
        else:
            peaks, area, sig, width = reject_peaks(x, y, baseline, signal)
            sf.integrate = peaks
            sf.stats = (area, sig, width)
            sf.save()

        # for p in peaks:
        #    pyplot.plot(p.time, p.orig, p.time, p.base)

        if sf.peaks is not None:
            peaks = sf.peaks
        else:
            peaks = resolve_peaks(peaks, area, sig, width)
            sf.peaks = peaks
            sf.save()

        # for p in peaks:
        #    pyplot.plot(p.time, p.orig, p.time, p.base)

        if ifprint:
            count = 0
            for p in peaks:
                print ("Peak {0}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}\t{4:0.1f}" +
                       "\t{5:0.1f}").format(count, p.retention, p.leading,
                                            p.tailing, p.area, p.height)
                count += 1

        pyplot.plot(x, y, "b", x, baseline, "y")
        for p in peaks:
            ilead = -1
            itail = -1
            for i in xrange(len(p.time)):
                if p.time[i] < p.leading:
                    ilead = i
                if itail < 0 and p.time[i] >= p.tailing:
                    itail = i
            if ilead < 0:
                ilead = 0
            pyplot.plot(p.time[ilead:itail], p.base[ilead:itail], "r")
            pyplot.axis([0, 94, -5000, 10000])

        return peaks

class Savefile_v01():
    CHECK_1 = "CHECK-BASELINE COMPLETE"
    CHECK_2 = "CHECK-INTEGRATE COMPLETE"
    CHECK_3 = "CHECK-PEAK LIST COMPLETE"

    def __init__(self, basefile):
        self.baseline = None
        self.integrate = None
        self.stats = None
        self.peaks = None
        fname = os.path.basename(basefile)
        self.filename = "_" + fname + FILE_SAVE

    def read(self):
        filename = self.filename
        if not filename.endswith(".sav"):
            raise ValueError("Incorrect file format (Requires .sav): " + filename)
        if not filename.endswith("_v01.sav"):
            raise ValueError("Incorrect version (Requires v01): " + filename)

        with open(filename, "r") as infile:
            integrate = False
            peaks = False
            temp = None
            count = 0
            xs = []
            ac = []
            lead = 0
            tail = 0
            peak = None
            for line in infile:
                splt = line.split("#")  # comment character
                line = splt[0].strip()
                if line == "":
                    continue
                if not integrate:
                    if line == Savefile_v01.CHECK_1:
                        self.baseline = temp
                        self.baseline.accel = np.array((xs, ac))
                        integrate = True
                        count = 0
                        xs = []
                        ac = []
                    else:
                        val, num = line.split(":")
                        if count == 0:
                            temp = Spline()
                            temp.y0 = float(num)
                        elif count == 1:
                            temp.d0 = float(num)
                        elif count > 1:
                            if val.startswith("x"):
                                xs.append(float(num))
                            elif val.startswith("a"):
                                ac.append(float(num))
                            else:
                                nums = num.split(",")
                                temp.xs = np.array([float(n) for n in nums])
                        count += 1
                elif not peaks:
                    if line == Savefile_v01.CHECK_2:
                        self.integrate = temp
                        peaks = True
                        count = 0
                    else:
                        if count == 0:
                            avg, avh, avw = line.split(",")
                            self.stats = (float(avg), float(avh), float(avw))
                            temp = []
                        else:
                            val, num = line.split(":")
                            nums = []
                            for s in num.split(","):
                                nums.append(float(s))
                            if val == "x":
                                peak = Peake()
                                peak.time = np.array(nums)
                            elif val == "y":
                                peak.orig = np.array(nums)
                            elif val == "b":
                                peak.base = np.array(nums)
                                peak.calculate()
                                temp.append(peak)
                                peak = None
                        count += 1
                else:
                    if line == Savefile_v01.CHECK_3:
                        self.peaks = temp
                        count = 0
                    else:
                        if count == 0:
                            temp = []
                        val, num = line.split(":")
                        nums = []
                        for s in num.split(","):
                            nums.append(float(s))
                        if val.startswith("P"):
                            peak = Peake()
                            lead = nums[1]
                            tail = nums[2]
                        elif val == "x":
                            peak.time = np.array(nums)
                        elif val == "y":
                            peak.orig = np.array(nums)
                        elif val == "b":
                            peak.base = np.array(nums)
                            peak.calculate(lead, tail)
                            temp.append(peak)
                            peak = None
                        count += 1
            infile.close()

    def save(self):
        filename = self.filename
        if os.path.exists(filename):
            os.remove(filename)
        if not self.filename.endswith(".sav"):
            raise ValueError("Incorrect file format (Saves as .sav): " + filename)
        if not self.filename.endswith("_v01.sav"):
            raise ValueError("Incorrect version (Saves as v01): " + filename)

        with open(filename, "w") as outfile:
            if self.baseline is not None:
                outfile.write("y0:" + str(self.baseline.y0) + "\n")
                outfile.write("d0:" + str(self.baseline.d0) + "\n")
                accel = self.baseline.accel
                for i in xrange(accel.shape[1]):
                    outfile.write("x" + str(i) + ":" + str(accel[0, i]) + "\n")
                    outfile.write("a" + str(i) + ":" + str(accel[1, i]) + "\n")
                outfile.write("times:")
                for i in xrange(len(self.baseline.xs)):
                    if i > 0:
                        outfile.write(",")
                    outfile.write(str(self.baseline.xs[i]))
                outfile.write("\n")
                outfile.write(Savefile_v01.CHECK_1 + "\n")
            if self.integrate is not None:
                outfile.write(str(self.stats[0]) + "," + str(self.stats[1]) + "," +
                              str(self.stats[2]) + "\n")
                for peak in self.integrate:
                    outfile.write("x:")
                    for i in xrange(len(peak.time)):
                        if i > 0:
                            outfile.write(",")
                        outfile.write(str(peak.time[i]))
                    outfile.write("\ny:")
                    for i in xrange(len(peak.orig)):
                        if i > 0:
                            outfile.write(",")
                        outfile.write(str(peak.orig[i]))
                    outfile.write("\nb:")
                    for i in xrange(len(peak.base)):
                        if i > 0:
                            outfile.write(",")
                        outfile.write(str(peak.base[i]))
                    outfile.write("\n")
                outfile.write(Savefile_v01.CHECK_2 + "\n")
            if self.peaks is not None:
                count = 0
                for p in self.peaks:
                    outfile.write("P" + str(count) + ":" +
                                  str(p.retention) + "," +
                                  str(p.leading) + "," +
                                  str(p.tailing) + "," +
                                  str(p.area) + "," +
                                  str(p.height) + "\n")
                    outfile.write("x:")
                    for i in xrange(len(p.time)):
                        if i > 0:
                            outfile.write(",")
                        outfile.write(str(p.time[i]))
                    outfile.write("\ny:")
                    for i in xrange(len(p.orig)):
                        if i > 0:
                            outfile.write(",")
                        outfile.write(str(p.orig[i]))
                    outfile.write("\nb:")
                    for i in xrange(len(p.base)):
                        if i > 0:
                            outfile.write(",")
                        outfile.write(str(p.base[i]))
                    outfile.write("\n")
                    count += 1
                outfile.write(Savefile_v01.CHECK_3 + "\n")
            outfile.close()

def testbounds(x, y):
    test = linear_interpolate(x, x[[0, -1]], y[[0, -1]])
    valmin = np.min(y - test)
    return valmin < 0

def peakbounds(x, y, le, ps, pe, leftlim):
    n = len(x)
    while le[ps] != -1 and ps > leftlim:
        ps -= 1
    while le[pe] != -1 and pe < n - 1:
        pe += 1
    failed = False
    while not failed:
        # test right
        ne = pe + 1
        if ne <= n - 1:
            while le[ne] != -1 and ne < n - 1:
                ne += 1
            failright = testbounds(x[ps:ne + 1], y[ps:ne + 1])
        else:
            failright = True
        if failright:
            ne = pe
        # test left
        ns = ps - 1
        if ns >= leftlim:
            while le[ns] != -1 and ns > leftlim:
                ns -= 1
            failleft = testbounds(x[ns:ne + 1], y[ns:ne + 1])
        else:
            failleft = True
        if failleft:
            ns = ps
        ps = ns
        pe = ne
        failed = failright and failleft
    return ps, pe

class Peak:

    def __init__(self):
        self.leading = 0
        self.retention = 0
        self.tailing = 0
        self.height = 0
        self.area = 0

def shimadzu_integrate(x, y, width=8, slope=1000, drift=800, tdbl=1000,
                       minarea=10000):
    da = derivative(x, y)
    le = local_extrema(x, y)
    NORM = "Normal"; PEAK = "Peak"
    base = y[0]
    state = NORM
    peakstart = 0
    forward = 0
    count = 0
    baseline = np.zeros_like(y)
    # driftonly = baseline * np.NaN
    for i in xrange(1, x.shape[0]):
        xold = x[i - 1]
        xval = x[i]
        yval = y[i]
        if i <= forward:
            base = baseline[i]
            continue
        if state == NORM and da[i - 1] >= slope:
            count += 1
            if count >= 3:
                count = 0
                state = PEAK
                peakstart = i
        else:
            count = 0
        if state == NORM:
            base = yval
        if state == PEAK:
            base += (xval - xold) * drift
            # driftonly[i] = base
            if yval < base:
                state = NORM
                i1, i2 = peakbounds(x, y, le, peakstart, i, forward)
                grabx = x[[i1, i2]]
                graby = y[[i1, i2]]
                fillx = x[i1:i2 + 1]
                baseline[i1:i2 + 1] = linear_interpolate(fillx, grabx, graby)
                base = baseline[i]
                forward = i2
        baseline[i] = base

    signal = y - baseline
    peaks = []
    rt = 0
    area = 0
    height = 0
    split = []
    state = NORM
    newbase = baseline * np.NaN
    for i in xrange(1, x.shape[0]):
        xold = x[i - 1]
        yold = signal[i - 1]
        xval = x[i]
        yval = signal[i]
        if yval > 0:
            if state != PEAK:
                state = PEAK
                peakstart = i - 1
            area += (yval + yold) * (xval - xold) / 2
            if yval > height:
                height = yval
                rt = xval
            if le[i] == -1:  # local minima
                if yval < height * 0.5:  # attempt to split
                    split.append(i)
        elif state == PEAK:
            state = NORM
            area *= 60  # convert to seconds
            if area >= minarea:
                peak = Peak()
                peak.area = area
                peak.height = height
                peak.leading = x[peakstart]
                peak.tailing = x[i]
                peak.retention = rt
                peaks.append(peak)
                newbase[peakstart:i] = baseline[peakstart:i]
            area = 0
            height = 0
            split = []
    return peaks, newbase

def fit(x, *p):
    scale = np.abs(p[0])
    shift1 = p[1]
    m1 = np.abs(p[2])
    shift2 = np.abs(p[3])
    m2 = np.abs(p[4])
    t1 = x - shift1
    t2 = x + shift2
    vals = scale * np.exp(-m1 * t1 ** 2) * np.exp(-m2 * t2)
    return vals

if __name__ == '__main__':
    pass

