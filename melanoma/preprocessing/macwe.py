import numpy as np
from scipy import ndimage
from itertools import cycle
from scipy.ndimage import binary_dilation, binary_erosion, \
                        gaussian_filter, gaussian_gradient_magnitude
class fcycle(object):

    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = self.funcs.next()
        return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0,1,0]]*3), np.flipud(np.eye(3)), np.rot90([[0,1,0]]*3)]
_P3 = [np.zeros((3,3,3)) for i in xrange(9)]

_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

_aux = np.zeros((0))
def SI(u):
    """SI operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"

    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)

    for i in xrange(len(P)):
        _aux[i] = binary_erosion(u, P[i])

    return _aux.max(0)

def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"

    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)

    for i in xrange(len(P)):
        _aux[i] = binary_dilation(u, P[i])

    return _aux.min(0)

# # SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

# Stopping factors (function g(I) in the paper).
def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""
    # The norm of the gradient.
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant')
    return 1.0/np.sqrt(1.0 + alpha*gradnorm)

def glines(img, sigma=1.0):
    """Stopping criterion for image black lines."""
    return gaussian_filter(img, sigma)

class MorphGAC(object):
    """Morphological GAC based on the Geodesic Active Contours."""

    def __init__(self, data, smoothing=1, threshold=0, balloon=0):
        self._u = None
        self._v = balloon
        self._theta = threshold
        self.smoothing = smoothing

        self.set_data(data)

    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0

    def set_balloon(self, v):
        self._v = v
        self._update_mask()

    def set_threshold(self, theta):
        self._theta = theta
        self._update_mask()

    def set_data(self, data):
        self._data = data
        self._ddata = np.gradient(data)
        self._update_mask()
        # The structure element for binary dilation and erosion.
        self.structure = np.ones((3,)*np.ndim(data))

    def _update_mask(self):
        """Pre-compute masks for speed."""
        self._threshold_mask = self._data > self._theta
        self._threshold_mask_v = self._data > self._theta/np.abs(self._v)

    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    data = property(lambda self: self._data,
                        set_data,
                        doc="The data that controls the snake evolution (the image or g(I)).")
    balloon = property(lambda self: self._v,
                        set_balloon)
    threshold = property(lambda self: self._theta,
                        set_threshold,
                        doc="The threshold value (0).")

    def step(self):
        """Perform a single step of the morphological snake evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        gI = self._data
        dgI = self._ddata
        theta = self._theta
        v = self._v

        if u is None:
            raise ValueError, "the levelset is not set (use set_levelset)"

        res = np.copy(u)

        # Balloon.
        if v > 0:
            aux = binary_dilation(u, self.structure)
        elif v < 0:
            aux = binary_erosion(u, self.structure)
        if v!= 0:
            res[self._threshold_mask_v] = aux[self._threshold_mask_v]

        # Image attachment.
        aux = np.zeros_like(res)
        dres = np.gradient(res)
        for el1, el2 in zip(dgI, dres):
            aux += el1*el2
        res[aux > 0] = 1
        res[aux < 0] = 0

        # Smoothing.
        for i in xrange(self.smoothing):
            res = curvop(res)

        self._u = res

    def run(self, iterations):
        """Run several iterations of the morphological snakes method."""
        for i in xrange(iterations):
            self.step()

class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""
    c01 = []

    def __init__(self, data, smoothing=1, lambda1=1, lambda2=1):
        """Create a Morphological ACWE solver.

        Parameters
        ----------
        data : ndarray
            The image data.
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter u.
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        """
        self._u = None
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.data = data

    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0

    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")

    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.

        u = self._u

        if u is None:
            raise ValueError, "the levelset function is not set (use set_levelset)"

        data = self.data

        # Determine c0 and c1.
        inside = u>0
        outside = u<=0

        c0 = data[outside].sum() / float(outside.sum())
        c1 = data[inside].sum() / float(inside.sum())

        # Image attachment.
        dres = np.array(np.gradient(u))
        abs_dres = np.abs(dres).sum(0)

        aux = abs_dres * (c0 - c1) * (c0 + c1 - 2*data)
        aux = abs_dres * (self.lambda1*np.square(data - c1) - self.lambda2*np.square(data - c0))

        res = np.copy(u)
        res[aux < 0] = 1
        res[aux > 0] = 0

        # Smoothing.
        for i in xrange(self.smoothing):
            res = curvop(res)

        self._u = res

    def run(self, iterations):
        """Run several iterations of the morphological Chan-Vese method."""
        for i in xrange(iterations):
            self.step()

def evolve(msnake, levelset=None, num_iters=20, background=None):
    from matplotlib import pyplot as ppl

    if levelset is not None:
        msnake.levelset = levelset

    for i in xrange(num_iters):
        msnake.step()

    return msnake.levelset

def circle_levelset(shape, center, sqradius,scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[map(slice, shape)].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def extract_lesion(img, start):
    macwe = MorphACWE(img, smoothing=4, lambda1=1, lambda2=1)
    macwe.levelset = circle_levelset(img.shape, (100, 170), 50)
    end_snake = evolve(macwe, num_iters=150, background=img)

    bool_end = end_snake.astype(bool)
    curr_bool = np.invert(bool_end)
    start[curr_bool] = 255
    print "[PRE-PROCESSING] Separated lesion from image"
    return start
