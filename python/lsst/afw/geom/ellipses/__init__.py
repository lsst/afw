#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""lsst.afw.geom.ellipses
"""
from __future__ import absolute_import
from .ellipsesLib import *

Separable = {
    (Distortion, DeterminantRadius):SeparableDistortionDeterminantRadius,
    (Distortion, TraceRadius):SeparableDistortionTraceRadius,
    (Distortion, LogDeterminantRadius):SeparableDistortionLogDeterminantRadius,
    (Distortion, LogTraceRadius):SeparableDistortionLogTraceRadius,
    (ConformalShear, DeterminantRadius):SeparableConformalShearDeterminantRadius,
    (ConformalShear, TraceRadius):SeparableConformalShearTraceRadius,
    (ConformalShear, LogDeterminantRadius):SeparableConformalShearLogDeterminantRadius,
    (ConformalShear, LogTraceRadius):SeparableConformalShearLogTraceRadius
}

#BaseCore.cast = lambda self: globals()[self.getName()].cast(self)

class EllipseMatplotlibInterface(object):
    """An interface for drawing the ellipse using matplotlib.

    This is typically initiated by calling Ellipse.plot(), which
    adds the interface as the matplotlib attribute of the ellipse
    object (this can be deleted later if desired).
    """

    def __init__(self, ellipse, scale=1.0, **kwds):
        import matplotlib.patches
        self.__ellipse = weakref.proxy(ellipse)
        self.scale = float(scale)
        core = Axes(self.__ellipse.getCore())
        core.scale(2.0 * scale)
        self.patch = matplotlib.patches.Ellipse(
            (self.__ellipse.getCenter().getX(), self.__ellipse.getCenter().getY()),
            core.getA(), core.getB(), core.getTheta() * 180.0 / numpy.pi,
            **kwds
            )

    def __getattr__(self, name):
        return getattr(self.patch, name)

    def update(self, show=True, rescale=True):
        """Update the matplotlib representation to the current ellipse parameters.
        """
        import matplotlib.patches
        core = _agl.Axes(self.__ellipse.getCore())
        core.scale(2.0 * scale)
        new_patch = matplotlib.patches.Ellipse(
            (self.__ellipse.getCenter().getX(), self.__ellipse.getCenter().getY()),
            core.a, core.b, core.theta * 180.0 / numpy.pi
            )
        new_patch.update_from(self.patch)
        axes = self.patch.get_axes()
        if axes is not None:
            self.patch.remove()
            axes.add_patch(new_patch)
        self.patch = new_patch
        if axes is not None:
            if rescale: axes.autoscale_view()
            if show: axes.figure.canvas.draw()

def Ellipse_plot(self, axes=None, scale=1.0, show=True, rescale=True, **kwds):
    """Plot the ellipse in matplotlib, adding a MatplotlibInterface
    object as the 'matplotlib' attribute of the ellipse.

    Aside from those below, keyword arguments for the
    matplotlib.patches.Patch constructor are also accepted
    ('facecolor', 'linestyle', etc.)

    Arguments:
    axes -------- A matplotlib.axes.Axes object, or None to use
    matplotlib.pyplot.gca().
    scale ------- Scale the displayed ellipse by this factor.
    show -------- If True, update the figure automatically.  Set
    to False for batch processing.
    rescale ----- If True, rescale the axes.
    """
    import matplotlib.pyplot
    self.matplotlib = self.MatplotlibInterface(self, scale, **kwds)
    if axes is None:
        axes = matplotlib.pyplot.gca()
    axes.add_patch(self.matplotlib.patch)
    if rescale: axes.autoscale_view()
    if show: axes.figure.canvas.draw()
    return self.matplotlib.patch


#Ellipse.MatplotlibInterface = EllipseMatplotlibInterface
#Ellipse.plot = Ellipse_plot
