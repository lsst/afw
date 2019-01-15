##########################
Introduction to CameraGeom
##########################

.. py:currentmodule:: lsst.afw.cameraGeom

.. _section_CameraGeom_Overview:

Overview
========

The cameraGeom package describes the geometry of an imaging camera, including the location of each detector (e.g. CCD) on the focal plane, information about the amplifier subregions of each detector, and the location of known bad pixels in each detector.
The cameraGeom package supports operations such as:

* Assemble images from raw data (combining amplifier subregions and trimming overscan).
  CameraGeom does not assemble an entire image (see :py:class:`lsst.ip.isr.AssembleCcdTask` for that) but includes functions in :py:mod:`assembleImage` that do much of the work.
* Transform 2-d points between various :ref:`camera coordinate systems <section_Camera_Coordinate_Systems>`,
  using :py:meth:`Camera.transform`.
  This can be used as part of generating a :py:class:`lsst.afw.geom.SkyWcs` or to examine the effects of optical distortion.
* Create a graphic showing the layout of detectors on the focal plane, using :py:func:`utils.plotFocalPlane`.

Data for constructing a Camera comes from the appropriate observatory-specific ``obs_`` package. For example ``obs_sdss`` contains data for the SDSS imager, and ``obs_subaru`` contains data for both Suprime-Cam and Hyper Suprime-Cam (HSC).

.. _section_Camera_Geometry_Utilities:

Camera Geometry Utilities
=========================

There are a few utilities available for visualizing and debugging Camera objects.
Examples of available utility methods are: display a particular amp, display an assembled sensor, display the full camera mosaic, plot the sensor boundaries with a grid of test points in :ref:`FOCAL_PLANE <CameraGeom_FOCAL_PLANE>` coordinates.
An example of how to use the utilities to visualize a camera is available in the obs_lsstSim package as ``$OBS_LSSTSIM_DIR/bin/displayCamera.py``.

.. _section_Camera_Coordinate_Systems:

Camera Coordinate Systems
=========================

The cameraGeom package supports the following camera-based 2-dimensional coordinate systems, and it is possible to add others:

.. _CameraGeom_FOCAL_PLANE:

``FOCAL_PLANE``
  Position on a 2-d planar approximation to the focal plane (x,y mm).
  The origin and orientation may be defined by the camera team, but we strongly recommend that the origin be on the optical axis and (if using CCD detectors) that the X axis be aligned along CCD rows.
  Note: location and orientation of detectors are defined in a 3-d version of ``FOCAL_PLANE`` coordinates
  (the z axis is also relevant).

.. _CameraGeom_FIELD_ANGLE:

``FIELD_ANGLE``
  Angle of a principal ray relative to the optical axis (x,y radians).
  The orientation of the x,y axes is the same as ``FOCAL_PLANE``.

.. _CameraGeom_PIXELS:

``PIXELS``
  Nominal position on the entry surface of a given detector (x, y unbinned pixels).
  For CCD detectors the x axis *must* be along rows (the direction of the serial register).
  This is required for our interpolation algorithm to interpolate across bad columns.

.. _CameraGeom_ACTUAL_PIXELS:

``ACTUAL_PIXELS``
  Like ``PIXELS``, but takes into account pixel-level distortions (deviations from the nominal model of uniformly spaced rectangular pixels).

.. _CameraGeom_TAN_PIXELS:

``TAN_PIXELS``
  Is a variant of ``PIXELS`` with estimated optical distortion removed.
  ``TAN_PIXELS`` is an affine transformation from ``FIELD_ANGLE`` coordinates, where ``PIXELS`` and ``TAN_PIXELS`` match at the center of the pupil frame.

.. _section_CameraGeom_Basic_Usage:

Basic Usage
===========

The file `examples/cameraGeomExample.py <https://github.com/lsst/afw/blob/master/examples/cameraGeomExample.py>`_ shows some basic usage of the cameraGeom package.

.. _section_CameraGeom_Objects:

Objects
=======

The cameraGeom package contains the following important objects; unless otherwise noted, all are available in both C++ and Python:

.. _subsection_CameraGeom_Camera:

Camera
------

A `Camera` is a collection of :ref:`Detectors <subsection_CameraGeom_Detector>`.

`Camera` also supports coordinate transformation between all :ref:`camera coordinate systems <section_Camera_Coordinate_Systems>`.

.. _subsection_CameraGeom_Detector:

Detector
--------

`Detector` contains information about a given imaging detector (typically a CCD), including its position and orientation in the focal plane and information about amplifiers (such as the image region, overscan and readout corner).
Amplifier data is stored as records in an :py:class:`lsst.afw.table.AmpInfoTable`, and `Detector` acts as a collection of :py:class:`lsst.afw.table.AmpInfoRecord`.

`Detector` also supports transformation between :ref:`FOCAL_PLANE <CameraGeom_FOCAL_PLANE>`, :ref:`PIXELS <CameraGeom_PIXELS>`, and (if a suitable transform has been provided) :ref:`ACTUAL_PIXELS <CameraGeom_ACTUAL_PIXELS>` coordinates.
However `Detector` does *not* support :ref:`FIELD_ANGLE <CameraGeom_FIELD_ANGLE>` coordinates; use a `Camera` for that.

.. _subsection_CameraGeom_CameraSys_and_CameraSysPrefix:

CameraSys and CameraSysPrefix
-----------------------------

`CameraSys` represents a :ref:`camera coordinate system <section_Camera_Coordinate_Systems>`. It contains
a coordinate system name and a detector name. The detector name is blank for non-detector-based
:ref:`camera coordinate systems <section_Camera_Coordinate_Systems>` such as
:ref:`FOCAL_PLANE <CameraGeom_FOCAL_PLANE>` and :ref:`FIELD_ANGLE <CameraGeom_FIELD_ANGLE>`,
but must always name a specific detector for detector-based coordinate systems.

`CameraSysPrefix` is a specialized variant of `CameraSys` that represents a detector-based coordinate system
when the detector is not specified. `CameraSysPrefix` contains a coordinate system name but no detector name.

A constant is provided each :ref:`camera coordinate system <section_Camera_Coordinate_Systems>`:

* __FOCAL_PLANE__ (a CoordSys) for the :ref:`FOCAL_PLANE <CameraGeom_FOCAL_PLANE>` system
* __FIELD_ANGLE__ (a CoordSys) for the :ref:`FIELD_ANGLE <CameraGeom_FIELD_ANGLE>` system
* __PIXELS__ (a CoordSysPrefix) for the :ref:`PIXELS <CameraGeom_PIXELS>` system
* __ACTUAL_PIXELS__ (a CoordSysPrefix) for the :ref:`ACTUAL_PIXELS <CameraGeom_ACTUAL_PIXELS>` system

All `Detector` methods that take a `CameraSys` also accept a `CameraSysPrefix` instead.
For example to transform a list of points from :ref:`PIXELS <CameraGeom_PIXELS>` to :ref:`FOCAL_PLANE <CameraGeom_FOCAL_PLANE>` system using a `Detector`:

.. code-block:: python

    focalPlanePoints = Detector.transform(pixelPoints, PIXELS, FOCAL_PLANE)

`Camera` methods always require a `CameraSys`; a `CameraSysPrefix` is not acceptable because the camera does not know which detector to use.
For example to transform a list of points from :ref:`PIXELS <CameraGeom_PIXELS>` on a specific detector to :ref:`FIELD_ANGLE <CameraGeom_FIELD_ANGLE>`:

.. code-block:: python

    fieldAnglePoints = camera.transform(pixelPoints, detector.makeCameraSys(PIXELS), FIELD_ANGLE)

.. _subsection_CameraGeom_TransformMap:

TransformMap
------------

`TransformMap` is a collection of :py:class:`lsst.afw.geom.TransformPoint2ToPoint2` "Transforms" from one :ref:`camera coordinate system <section_Camera_Coordinate_Systems>` to another.

`Camera` and `Detector` both contain `TransformMaps`.
The transform map in Camera does not support detector-based coordinate systems (e.g. :ref:`PIXELS <CameraGeom_PIXELS>`), but `Camera.getTransform` and `Camera.transform` do support detector-based coordinate systems (since the camera contains information about the detectors).
