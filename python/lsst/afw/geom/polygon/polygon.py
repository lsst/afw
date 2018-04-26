from ._polygon import Polygon

__all__ = []  # import this module only for its side effects


def __repr__(self):
    return "%s(%s)" % (self.__class__.__name__, [p for p in self.getVertices()])


def __reduce__(self):
    return self.__class__, (self.getVertices(),)


def __iter__(self):
    """Iterator over vertices"""
    vertices = self.getVertices()
    return iter(vertices)


def __getitem__(self, i):
    return [pt for pt in self][i]


def __len__(self):
    return self.getNumEdges()


def __contains__(self, point):
    """point in polygon?"""
    return self.contains(point)


def display(self, xy0=None, frame=1, ctype=None):
    """Display polygon on existing frame in ds9"""
    import lsst.afw.geom as afwGeom
    import lsst.afw.display.ds9 as ds9
    xy0 = afwGeom.Extent2D(0, 0) if xy0 is None else afwGeom.Extent2D(xy0)
    with ds9.Buffering():
        for p1, p2 in self.getEdges():
            ds9.line((p1 - xy0, p2 - xy0), frame=frame, ctype=ctype)


def plot(self, axes=None, **kwargs):
    """Plot polygon with matplotlib

    Parameters
    ----------
    axes : `matplotlib.axes.Axes`
        Matplotlib axes to use, or None
    kwargs : any
        Additional arguments to `matplotlib.axes.Axes.plot`
        or `matplotlib.axes.Axes.scatter`.

    Returns
    -------
    axes : `matplotlib.axes.Axes`
        The axes used to make the plot (same as ``axes``, if provided).
    """
    import numpy
    if axes is None:
        import matplotlib.pyplot as plt
        plt.figure()
        axes = plt.axes()
    for p1, p2 in self.getEdges():
        x = (p1.getX(), p2.getX())
        y = (p1.getY(), p2.getY())
        axes.plot(x, y, **kwargs)
    vertices = self.getVertices()
    x = numpy.array([p[0] for p in vertices])
    y = numpy.array([p[1] for p in vertices])
    axes.scatter(x, y, **kwargs)
    return axes


Polygon.__repr__ = __repr__
Polygon.__reduce__ = __reduce__
Polygon.__iter__ = __iter__
Polygon.__getitem__ = __getitem__
Polygon.__len__ = __len__
Polygon.__contains__ = __contains__
Polygon.display = display
Polygon.plot = plot
