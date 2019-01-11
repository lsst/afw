# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .polygon import Polygon


@continueClass  # noqa F811
class Polygon:
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
        """Display polygon on existing frame"""
        import lsst.geom
        import lsst.afw.display as afwDisplay
        xy0 = lsst.geom.Extent2D(0, 0) if xy0 is None else lsst.geom.Extent2D(xy0)
        disp = afwDisplay.Display(frame=frame)
        with disp.Buffering():
            for p1, p2 in self.getEdges():
                disp.line((p1 - xy0, p2 - xy0), ctype=ctype)

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
