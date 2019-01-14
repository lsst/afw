.. py:currentmodule:: lsst.afw.math

.. _lsst.afw.math:

#############
lsst.afw.math
#############

Mathematical functions such as convolution and image statistics

Key features:

* Function objects `FunctionF` and `FunctionD`
* `Statistics`
* `Background` estimation of images.
* `Interpolate`
* `Kernel`
* Optimization of functions is handled by `minimize()`
* Convolution of images is handled by `convolve()`
* Warping of images is handled by `warpImage()`
* Manipulating spatially-distributed sets of objects (e.g. PSF candidates)

.. _lsst.afw.math-using:

Using lsst.afw.math
===================

.. toctree::
   :maxdepth: 1

   Background-example
   SpatialCellSet-example
   Statistics-example

.. _lsst.afw.math-contributing:

Contributing
============

``lsst.afw`` is developed at https://github.com/lsst/afw.
You can find Jira issues for this module under the `afw <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20afw>`_ component.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

Python API reference
====================

.. automodapi:: lsst.afw.math
   :no-main-docstr:
   :skip: Persistable

.. lsst.afw.table.io.Persistable is lifted into lsst.afw.math.mathLib for reasons unknown
