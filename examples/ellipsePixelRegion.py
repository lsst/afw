#!/usr/bin/env python

import lsst.afw.geom.ellipses
import numpy
from matplotlib import pyplot


def main():
    axes = lsst.afw.geom.ellipses.Axes(4, 3, 1)
    ellipse = lsst.afw.geom.ellipses.Ellipse(
        axes, lsst.afw.geom.Point2D(0.25338, 0.76032))
    region = lsst.afw.geom.ellipses.PixelRegion(ellipse)
    for bbox in [ellipse.computeBBox(), lsst.afw.geom.Box2D(region.getBBox())]:
        corners = bbox.getCorners()
        pyplot.fill([p.getX() for p in corners], [p.getY()
                                                  for p in corners], alpha=0.2)
    envelope = region.getBBox()
    envelope.grow(2)
    ellipse.plot(alpha=0.2)
    ellX = []
    ellY = []
    allX, allY = numpy.meshgrid(
        numpy.arange(envelope.getBeginX(), envelope.getEndX()),
        numpy.arange(envelope.getBeginY(), envelope.getEndY())
    )
    gt = ellipse.getGridTransform()
    mgt = gt.getMatrix()
    transX = mgt[0, 0] * allX + mgt[0, 1] * allY + mgt[0, 2]
    transY = mgt[1, 0] * allX + mgt[1, 1] * allY + mgt[1, 2]
    allR = (transX**2 + transY**2)**0.5
    pyplot.plot(ellX, ellY, 'ro', markeredgewidth=0, alpha=0.5)
    pyplot.plot(allX[allR < 1], allY[allR < 1], '+')
    pyplot.plot(allX[allR > 1], allY[allR > 1], 'x')


if __name__ == "__main__":
    main()
    pyplot.show()
