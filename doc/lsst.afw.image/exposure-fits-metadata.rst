.. :py:currentmodule:: lsst.afw.image

##########################################
FITS metadata for lsst.afw.image Exposures
##########################################

`ExposureInfo` and contents such as `Calib` and `VisitInfo` can be stored as FITS header keywords.
The catalog schemas are described in ``ExposureInfo.cc``, ``Calib.cc``, and ``VisitInfo.cc``.
This page describes the equivalent FITS header keywords.

HDU 0
=====

.. list-table::
   :widths: 2 1 1 3 5
   :header-rows: 1

   * - Keyword
     - Format
     - Units
     - LSST object and accessor [1]_
     - Description

   * - ``DETNAME``
     - str
     -
     - `Detector.getAmpInfo`
     - detector name

   * - ``DETSER``
     - str
     -
     - `Detector.getAmpInfo`
     - detector serial number

   * - ``EXPID``
     - long int
     -
     - `VisitInfo.getExposureId`
     - Exposure ID

   * - ``EXPTIME``
     - float
     - sec
     - `VisitInfo.getExposureTime`
     - Exposure duration (shutter open time)

   * - ``DARKTIME``
     - float
     - sec
     - `VisitInfo.getDarkTime`
     - Time from CCD flush to readout, including shutter open time (despite the name)

   * - ``DATE-AVG``
     - ISO8601
     -
     - `VisitInfo.getDate`
     - Date at middle of exposure, at boresight. No time zone character allowed.

   * - ``TIMESYS``
     - str
     -
     -
     - Must exist and must be set to ``"TAI"`` if ``DATE-AVG`` is present.

   * - ``TIME-MID``
     - ISO8601
     -
     - `VisitInfo.getDate`
     - Deprecated; read if ``DATE-AVG`` not found, for backwards compatibility.
       Date at middle of exposure. The time zone character must exist and must be ``"Z"``.
       Note: ``TIME-MID`` is always UTC (even if the comment claims it is TAI) and ``TIMESYS`` is ignored.

   * - ``MJD-AVG-UT1``
     - float
     - MJD
     - `VisitInfo.getUt1`
     - UT1 date middle of exposure, at boresight

   * - ``AVG-ERA``
     - float
     - deg
     - `VisitInfo.getEra`
     - Earth rotation angle at middle of exposure, at boresight.

   * - ``BORE-RA``
     - float
     - deg
     - `VisitInfo.getBoreRaDec`
     - Position of boresight, ICRS RA.

   * - ``BORE-DEC``
     - float
     - deg
     - `VisitInfo.getBoreRaDec`
     - Position of boresight, ICRS Dec.

   * - ``BORE-AZ``
     - float
     - deg
     - `VisitInfo.getBoreAzAlt`
     - Position of boresight, refracted apparent topocentric Az.

   * - ``BORE-ALT``
     - float
     - deg
     - `VisitInfo.getBoreAzAlt`
     - Position of boresight, refracted apparent topocentric Alt.

   * - ``BORE-ROTANG``
     - float
     - deg
     - `VisitInfo.getBoreRotAngle`
     - Orientation of rotator at boresight.

   * - ``ROTTYPE``
     - str
     -
     - `VisitInfo.getRotType`
     - Type of rotation; one of:

       - ``UNKNOWN``
       - ``SKY``: position angle of focal plane +Y measured from N through E.
       - ``HORIZON``: position angle of focal plane +Y measured from +Alt through +Az.
       - ``MOUNT``: the position sent to the instrument rotator; the details depend on the rotator.

   * - ``OBS-LONG``
     - float
     - deg
     - `VisitInfo.getObservatory`
     - Longitude of telescope.

   * - ``OBS-LAT``
     - float
     - deg
     - `VisitInfo.getObservatory`
     - Latitude of telescope (positive eastward).

   * - ``OBS-ELEV``
     - float
     - m
     - `VisitInfo.getObservatory`
     - Geodetic elevation of telescope (meters above reference spheroid).

   * - ``AIRTEMP``
     - float
     - C
     - `VisitInfo.getWeather`
     - Air temperature.

   * - ``AIRPRESS``
     - float
     - Pascals
     - `VisitInfo.getWeather`
     - Air pressure.

   * - ``HUMIDITY``
     - float
     - %
     - `VisitInfo.getWeather`
     - Relative humidity.

   * - ``FLUXMAG0``
     - float
     - ADU
     - `Calib.getFluxMag0`
     - Flux of a zero-magnitude object.

   * - ``FLUXMAG0ERR``
     - float
     - ADU
     - `Calib.getFluxMag0Err`
     - Error in the flux of a zero-magnitude object.

   * - ``FILTER``
     - str
     -
     - `Filter.getName`
     - Name of filter.

   * - ``AR_HDU``
     - int
     -
     -
     - HDU containing the archive used to store ancillary objects

   * - ``COADD_INPUTS_ID``
     - int
     -
     -
     - Archive ID for coadd inputs catalogs

   * - ``AP_CORR_MAP_ID``
     - int
     -
     -
     - Archive ID for aperture correction map

   * - ``PSF_ID``
     - int
     -
     -
     - Archive ID for the Exposure's main Psf

   * - ``WCS_ID``
     - int
     -
     -
     - Archive ID for the Exposure's main Wcs

   * - ``VALID_POLYGON_ID``
     - int
     -
     -
     - Archive ID for the Exposure's valid polygon

HDUs 1 to 3
===========

.. list-table::
   :widths: 2 1 1 3 5
   :header-rows: 1

   * - Keyword
     - Format
     - Units
     - LSST object and accessor [1]_
     - Description

   * - ``LTV1``
     - int
     -
     - `Exposure.getXY0`
     - Image origin, X axis = -_x0 [2]_.

   * - ``LTV2``
     - int
     -
     - `Exposure.getXY0`
     - Image origin, Y axis = -_y0 [2]_.

.. [1] Unless otherwise noted, each object is contained in the ExposureInfo and has a getter.
   Thus to get ``VisitInfo`` use ``exposure.getExposureInfo().getVisitInfo()``.
   In some cases a direct shortcut is also available, e.g. ``exposure.getFilter()`` is a shortcut for ``exposure.getExposureInfo().getFilter()``.

.. [2] If this Exposure is a portion of a larger image, ``_x0`` and ``_y0`` indicate the origin (the position of the bottom left corner) of the sub-image with respect to the origin of the parent image.
   This is stored in the fits header using the LTV convention used by STScI (see `§2.6.2 of HST Data Handbook for STIS`_, version 5.0).
   This is not a FITS standard keyword, but is recognised by ds9.
   LTV keywords use the opposite convention to the LSST, in that they represent the position of the origin of the parent image relative to the origin of the sub-image.

.. _`§2.6.2 of HST Data Handbook for STIS`: http://www.stsci.edu/hst/stis/documents/handbooks/currentDHB/ch2_stis_data7.html#429287
