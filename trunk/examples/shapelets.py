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
            basis.fillEvaluation(z[i,j,:], float(px), float(py))
    return z

def compareMoments(function, x, y, z):
    e = function.evaluate()
    monopole = z.sum()
    dipole = geom.Point2D((x * z).sum() / monopole, (y * z).sum() / monopole)
    dx = x - dipole.getX()
    dy = y - dipole.getY()
    quadrupole = ellipses.Quadrupole(
        (dx**2 * z).sum() / monopole,
        (dy**1 * z).sum() / monopole,
        (dx * dy * z).sum() / monopole
        )
    print ellipses.Ellipse(quadrupole, monopole)
    print e.computeMoments()

def checkIntegration(basis, x, y, z):
    d = (x[1:] - x[:-1]).mean() * (y[1:] - x[:-1]).mean()
    array1 = numpy.zeros(shapelets.computeSize(basis.getOrder()), dtype=float)
    basis.fillIntegration(array1)
    array2 = z.sum(axis=0).sum(axis=0) * d
    print "integration equal:", numpy.abs(array1 - array2).max() < 0.005

def plotBasisImages(basis, z):
    n = basis.getOrder()
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

def processBasis(basis, x, y):
    z = makeBasisImages(basis, x, y)
    plotBasisImages(basis, z)
    checkIntegration(basis, x, y, z)


def main():
    x = numpy.linspace(-5, 5, 101)
    y = numpy.linspace(-5, 5, 101)
    ellipse = ellipses.Quadrupole(ellipses.Axes(1.2, 0.8, 0.3))
    hermiteBasis = shapelets.BasisEvaluator(4, shapelets.HERMITE)
    laguerreBasis = shapelets.BasisEvaluator(4, shapelets.LAGUERRE)
    processBasis(hermiteBasis, x, y)
    processBasis(laguerreBasis, x, y)
    pyplot.show()

if __name__ == "__main__":
    numpy.set_printoptions(suppress=True)
    main()
