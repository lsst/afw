from lsst.afw.math import shapelets
from lsst.afw import geom
from lsst.afw.geom import ellipses
from matplotlib import pyplot
from matplotlib import ticker
import numpy

def makeBasisImages(basis, x, y):
    z = numpy.zeros((y.size, x.size, shapelets.computeSize(basis.getOrder())), dtype=float)
    for i, py in enumerate(y):
        for j, px in enumerate(x):
            basis.fillEvaluationVector(z[i,j,:], float(px), float(py))
    return z

def plotBasisImages(basis, x, y):
    n = basis.getOrder()
    z = makeBasisImages(basis, x, y)
    k = 0
    vmin = z.min()
    vmax = z.max()
    pyplot.figure()
    for i in range(n+1):
        for j in range(i+1):
            axes = pyplot.subplot(n+1, n+1, (n+1) * i + j + 1)
            axes.imshow(z[:,:,k], interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
            axes.yaxis.set_major_locator(ticker.NullLocator())
            axes.xaxis.set_major_locator(ticker.NullLocator())
            if basis.getBasisType() == shapelets.HERMITE:
                pyplot.xlabel("x=%d, y=%d" % (j, i-j))
            else:
                pyplot.xlabel("p=%d, q=%d (%s)" % (i-j/2, j/2, "Im" if j % 2 else "Re"))
            k += 1

def main():
    x = numpy.linspace(-5, 5, 101)
    y = numpy.linspace(-5, 5, 101)
    ellipse = ellipses.Quadrupole(ellipses.Axes(1.2, 0.8, 0.3))
    unitHermiteBasis = shapelets.UnitShapeletBasis(4, shapelets.HERMITE)
    unitLaguerreBasis = shapelets.UnitShapeletBasis(4, shapelets.LAGUERRE)
    ellipticalHermiteBasis = shapelets.EllipticalShapeletBasis(4, shapelets.HERMITE, ellipse)
    ellipticalLaguerreBasis = shapelets.EllipticalShapeletBasis(4, shapelets.LAGUERRE, ellipse)
    plotBasisImages(unitHermiteBasis, x, y)
    plotBasisImages(unitLaguerreBasis, x, y)
    plotBasisImages(ellipticalHermiteBasis, x, y)
    plotBasisImages(ellipticalLaguerreBasis, x, y)
    pyplot.show()

if __name__ == "__main__":
    main()
