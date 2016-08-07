#!/usr/bin/env python

import timeit
import sys
import numpy
from matplotlib import pyplot
from lsst.afw.math import LeastSquares

nDataList = 2**numpy.arange(4, 14, 2, dtype=int)
dimensionList = 2**numpy.arange(1, 8, dtype=int)

def run(number=10):
    setup = """
import numpy
from lsst.afw.math import LeastSquares
nData = %d
dimension = %d
data = numpy.random.randn(nData)
design = numpy.random.randn(dimension, nData).transpose()
"""
    statement = "LeastSquares.fromDesignMatrix(design, data, LeastSquares.%s).solve()"
    results = {}
    totalCount = 0
    for nData in nDataList:
        for dimension in dimensionList:
            if dimension <= nData:
                totalCount += 1
    totalCount *= 3
    progress = 0
    sys.stderr.write("Timing: ")
    for method in ("DIRECT_SVD", "NORMAL_EIGENSYSTEM", "NORMAL_CHOLESKY"):
        results[method] = numpy.zeros(nDataList.shape + dimensionList.shape, dtype=float)
        for i, nData in enumerate(nDataList):
            for j, dimension in enumerate(dimensionList):
                if dimension <= nData:
                    results[method][i,j] = timeit.timeit(statement % method, setup % (nData, dimension),
                                                         number = number) / number
                    progress += 1
                else:
                    results[method][i,j] = float("NaN")
            sys.stderr.write("%0.2f " % (float(progress) / totalCount))
    sys.stderr.write("\n")
    return results

def plot(results):
    colors = {"DIRECT_SVD": "r", "NORMAL_EIGENSYSTEM": "g", "NORMAL_CHOLESKY": "b"}
    pyplot.figure()
    alpha = 0.75
    for method in results:
        for i, nData in enumerate(nDataList):
            if i == len(nDataList)-1:
                label = method
            else:
                label = "_nolegend_"
            pyplot.loglog(dimensionList, results[method][i,:], colors[method], alpha=alpha, label=label,
                          marker="o", markeredgewidth=0, markersize=3)
            j = -1
            while numpy.isnan(results[method][i,j]): j -= 1
            pyplot.text(dimensionList[j], results[method][i,j], " %d" % nData,
                        color=colors[method], size="x-small")
    pyplot.xlabel("# of parameters (# of data points in labels)")
    pyplot.ylabel("time (s)")
    pyplot.legend(loc="upper left")

    pyplot.figure()
    for method in results:
        for j, dimension in enumerate(dimensionList):
            if j == len(dimensionList)-1:
                label = method
            else:
                label = "_nolegend_"
            pyplot.loglog(nDataList, results[method][:,j], colors[method], alpha=alpha, label=label,
                          marker="o", markeredgewidth=0, markersize=3)
            pyplot.text(nDataList[i], results[method][-1,j], " %d" % dimension,
                        color=colors[method], size="x-small")
    pyplot.xlabel("# of data points (# of parameters in labels)")
    pyplot.ylabel("time (s)")
    pyplot.legend(loc="upper left")

    pyplot.show()

def main():
    results = run()
    plot(results)


if __name__ == "__main__":
    main()
