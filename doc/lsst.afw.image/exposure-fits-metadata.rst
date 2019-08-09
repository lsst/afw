.. :py:currentmodule:: lsst.afw.image

##########################################
FITS metadata for lsst.afw.image Exposures
##########################################

`ExposureInfo` and contents such as `Filter` and `VisitInfo` can be stored as FITS header keywords.
The catalog schemas are described in ``ExposureInfo.cc``, ``Filter.cc``, and ``VisitInfo.cc``.
This page describes the equivalent FITS header keywords.

HDU 0
=====

.. list-table::
   :widths: 2 1 1 5
   :header-rows: 1

   * - Keyword
     - Format
     - Units
     - Accessor [1]_ and description

   * - ``DETNAME``
     - str
     -
     - `Detector.getAmpInfo`

       Detector name.

   * - ``DETSER``
     - str
     -
     - `Detector.getAmpInfo`

       Detector serial number.

   * - ``EXPID``
     - long int
     -
     - `VisitInfo.getExposureId`

       Exposure ID.

   * - ``EXPTIME``
     - float
     - sec
     - `VisitInfo.getExposureTime`

       Exposure duration (shutter open time).

   * - ``DARKTIME``
     - float
     - sec
     - `VisitInfo.getDarkTime`

       Time from CCD flush to readout, including shutter open time (despite the name)

   * - ``DATE-AVG``
     - ISO8601
     -
     - `VisitInfo.getDate`

       Date at middle of exposure, at boresight. No time zone character allowed.

   * - ``TIMESYS``
     - str
     -
     - Must exist and must be set to ``"TAI"`` if ``DATE-AVG`` is present.

   * - ``TIME-MID``
     - ISO8601
     -
     - `VisitInfo.getDate`

       Deprecated; read if ``DATE-AVG`` not found, for backwards compatibility.
       Date at middle of exposure. The time zone character must exist and must be ``"Z"``.
       Note: ``TIME-MID`` is always UTC (even if the comment claims it is TAI) and ``TIMESYS`` is ignored.

   * - ``MJD-AVG-UT1``
     - float
     - MJD
     - `VisitInfo.getUt1`

       UT1 date middle of exposure, at boresight

   * - ``AVG-ERA``
     - float
     - deg
     - `VisitInfo.getEra`

       Earth rotation angle at middle of exposure, at boresight.

   * - ``BORE-RA``
     - float
     - deg
     - `VisitInfo.getBoreRaDec`

       Position of boresight, ICRS RA.

   * - ``BORE-DEC``
     - float
     - deg
     - `VisitInfo.getBoreRaDec`

       Position of boresight, ICRS Dec.

   * - ``BORE-AZ``
     - float
     - deg
     - `VisitInfo.getBoreAzAlt`

       Position of boresight, refracted apparent topocentric Az.

   * - ``BORE-ALT``
     - float
     - deg
     - `VisitInfo.getBoreAzAlt`

       Position of boresight, refracted apparent topocentric Alt.

   * - ``BORE-ROTANG``
     - float
     - deg
     - `VisitInfo.getBoreRotAngle`

       Orientation of rotator at boresight.

   * - ``ROTTYPE``
     - str
     -
     - `VisitInfo.getRotType`

       Type of rotation; one of:

       - ``UNKNOWN``
       - ``SKY``: position angle of focal plane +Y measured from N through E.
       - ``HORIZON``: position angle of focal plane +Y measured from +Alt through +Az.
       - ``MOUNT``: the position sent to the instrument rotator; the details depend on the rotator.

   * - ``OBS-LONG``
     - float
     - deg
     - `VisitInfo.getObservatory`
       Longitude of telescope.

   * - ``OBS-LAT``
     - float
     - deg
     - `VisitInfo.getObservatory`

       Latitude of telescope (positive eastward).

   * - ``OBS-ELEV``
     - float
     - m
     - `VisitInfo.getObservatory`

       Geodetic elevation of telescope (meters above reference spheroid).

   * - ``AIRTEMP``
     - float
     - C
     - `VisitInfo.getWeather`

       Air temperature.

   * - ``AIRPRESS``
     - float
     - Pascals
     - `VisitInfo.getWeather`

       Air pressure.

   * - ``HUMIDITY``
     - float
     - %
     - `VisitInfo.getWeather`

       Relative humidity.

   * - ``FILTER``
     - str
     -
     - `Filter.getName`

       Name of filter.

   * - ``AR_HDU``
     - int
     -
     - HDU containing the archive used to store ancillary objects

   * - ``COADD_INPUTS_ID``
     - int
     -
     - Archive ID for coadd inputs catalogs

   * - ``AP_CORR_MAP_ID``
     - int
     -
     - Archive ID for aperture correction map

   * - ``PSF_ID``
     - int
     -
     - Archive ID for the Exposure's main Psf

   * - ``WCS_ID``
     - int
     -
     - Archive ID for the Exposure's main Wcs

   * - ``VALID_POLYGON_ID``
     - int
     -
     - Archive ID for the Exposure's valid polygon

   * - ``ARCHIVE_ID_[name]``
     - int
     -
     - `getComponent("[name]")`

       Archive ID for an arbitrary Exposure component

HDUs 1 to 3
===========

.. list-table::
   :widths: 2 1 1 5
   :header-rows: 1

   * - Keyword
     - Format
     - Units
     - Accessor [1]_ and description

   * - ``CRPIX1A``
     - double
     - pixels
     - X axis reference pixel, always ``1.0``.

   * - ``CRPIX2A``
     - double
     - pixels
     - Y axis reference pixel, always ``1.0``.

   * - ``CRVAL1A``
     - double
     -
     - `Exposure.getXY0`

       Access as ``exposure.getXY0[0]``.

   * - ``CRVAL2A``
     - double
     -
     - `Exposure.getXY0`

       Access as ``exposure.getXY0[1]``.

   * - ``CTYPE1A``
     - str
     -
     - Always ``"LINEAR"``.

   * - ``CTYPE2A``
     - str
     -
     - Always ``"LINEAR"``.

   * - ``CUNIT1A``
     - str
     -
     - Always ``"PIXEL"``.

   * - ``CUNIT2A``
     - str
     -
     - Always ``"PIXEL"``.

.. [1] Unless otherwise noted, each object is contained in the ExposureInfo and has a getter.
   Thus to get ``VisitInfo`` use ``exposure.getExposureInfo().getVisitInfo()``.
   In some cases a direct shortcut is also available, e.g. ``exposure.getFilter()`` is a shortcut for ``exposure.getExposureInfo().getFilter()``.
