#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
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
import math
import os
import unittest
import collections
import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.daf.base import DateTime, PropertySet, PropertyList
from lsst.geom import Angle, degrees, SpherePoint
from lsst.afw.coord import Observatory, Weather
import lsst.afw.image as afwImage

RotTypeEnumNameDict = {
    afwImage.RotType.UNKNOWN: "UNKNOWN",
    afwImage.RotType.SKY: "SKY",
    afwImage.RotType.HORIZON: "HORIZON",
    afwImage.RotType.MOUNT: "MOUNT",
}


def propertySetFromDict(keyValDict):
    """Make an lsst.daf.base.PropertySet from a dict of key: value"""
    metadata = PropertySet()
    for key, val in keyValDict.items():
        metadata.set(key, val)
    return metadata


def makeVisitInfo(data):
    """Return a VisitInfo constructed from a VisitInfoData namedtuple."""
    return afwImage.VisitInfo(data.exposureId,
                              data.exposureTime,
                              data.darkTime,
                              data.date,
                              data.ut1,
                              data.era,
                              data.boresightRaDec,
                              data.boresightAzAlt,
                              data.boresightAirmass,
                              data.boresightRotAngle,
                              data.rotType,
                              data.observatory,
                              data.weather,
                              data.instrumentLabel,
                              data.id,
                              data.focusZ,
                              )


class VisitInfoTestCase(lsst.utils.tests.TestCase):
    """Test lsst.afw.image.VisitInfo, a simple struct-like class"""

    def setUp(self):
        self.testDir = os.path.dirname(__file__)

        def computeLstHA(data):
            """Return LST, Hour Angle, computed from VisitInfoData."""
            localEra = data.era + data.observatory.getLongitude()
            hourAngle = localEra - data.boresightRaDec[0]
            return localEra, hourAngle

        fields = ['exposureId',
                  'exposureTime',
                  'darkTime',
                  'date',
                  'ut1',
                  'era',
                  'boresightRaDec',
                  'boresightAzAlt',
                  'boresightAirmass',
                  'boresightRotAngle',
                  'rotType',
                  'observatory',
                  'weather',
                  'instrumentLabel',
                  'id',
                  'focusZ',
                  ]
        VisitInfoData = collections.namedtuple("VisitInfoData", fields)
        data1 = VisitInfoData(exposureId=10313423,
                              exposureTime=10.01,
                              darkTime=11.02,
                              date=DateTime(
                                  65321.1, DateTime.MJD, DateTime.TAI),
                              ut1=12345.1,
                              era=45.1*degrees,
                              boresightRaDec=SpherePoint(
                                  23.1*degrees, 73.2*degrees),
                              boresightAzAlt=SpherePoint(
                                  134.5*degrees, 33.3*degrees),
                              boresightAirmass=1.73,
                              boresightRotAngle=73.2*degrees,
                              rotType=afwImage.RotType.SKY,
                              observatory=Observatory(
                                  11.1*degrees, 22.2*degrees, 0.333),
                              weather=Weather(1.1, 2.2, 34.5),
                              instrumentLabel="TestCameraOne",
                              id=987654,
                              focusZ=1.5,
                              )
        self.data1 = data1
        self.localEra1, self.hourAngle1 = computeLstHA(data1)
        data2 = VisitInfoData(exposureId=1,
                              exposureTime=15.5,
                              darkTime=17.8,
                              date=DateTime(
                                  55321.2, DateTime.MJD, DateTime.TAI),
                              ut1=312345.1,
                              era=25.1*degrees,
                              boresightRaDec=SpherePoint(
                                  2.1*degrees, 33.2*degrees),
                              boresightAzAlt=SpherePoint(13.5*degrees, 83.3*degrees),
                              boresightAirmass=2.05,
                              boresightRotAngle=-53.2*degrees,
                              rotType=afwImage.RotType.HORIZON,
                              observatory=Observatory(
                                  22.2*degrees, 33.3*degrees, 0.444),
                              weather=Weather(2.2, 3.3, 44.4),
                              instrumentLabel="TestCameraTwo",
                              id=123456,
                              focusZ=-1.5,
                              )
        self.data2 = data2
        self.localEra2, self.hourAngle2 = computeLstHA(data2)

    def _testValueConstructor(self, data, localEra, hourAngle):
        visitInfo = makeVisitInfo(data)
        with self.assertWarns(FutureWarning):
            self.assertEqual(visitInfo.getExposureId(), data.exposureId)
        self.assertEqual(visitInfo.getExposureTime(), data.exposureTime)
        self.assertEqual(visitInfo.getDarkTime(), data.darkTime)
        self.assertEqual(visitInfo.getDate(), data.date)
        self.assertEqual(visitInfo.getUt1(), data.ut1)
        self.assertEqual(visitInfo.getEra(), data.era)
        self.assertEqual(visitInfo.getBoresightRaDec(), data.boresightRaDec)
        self.assertEqual(visitInfo.getBoresightAzAlt(), data.boresightAzAlt)
        self.assertEqual(visitInfo.getBoresightAirmass(),
                         data.boresightAirmass)
        self.assertEqual(visitInfo.getBoresightRotAngle(),
                         data.boresightRotAngle)
        self.assertEqual(visitInfo.getRotType(), data.rotType)
        self.assertEqual(visitInfo.getObservatory(), data.observatory)
        self.assertEqual(visitInfo.getInstrumentLabel(), data.instrumentLabel)
        self.assertEqual(visitInfo.getWeather(), data.weather)
        self.assertEqual(visitInfo.getLocalEra(), localEra)
        self.assertEqual(visitInfo.getBoresightHourAngle(), hourAngle)
        self.assertEqual(visitInfo.getId(), data.id)
        self.assertEqual(visitInfo.getFocusZ(), data.focusZ)

    def _testProperties(self, data, localEra, hourAngle):
        """Test property attribute accessors."""
        visitInfo = makeVisitInfo(data)
        self.assertEqual(visitInfo.exposureTime, data.exposureTime)
        self.assertEqual(visitInfo.darkTime, data.darkTime)
        self.assertEqual(visitInfo.date, data.date)
        self.assertEqual(visitInfo.ut1, data.ut1)
        self.assertEqual(visitInfo.era, data.era)
        self.assertEqual(visitInfo.boresightRaDec, data.boresightRaDec)
        self.assertEqual(visitInfo.boresightAzAlt, data.boresightAzAlt)
        self.assertEqual(visitInfo.boresightAirmass, data.boresightAirmass)
        self.assertEqual(visitInfo.boresightRotAngle, data.boresightRotAngle)
        self.assertEqual(visitInfo.rotType, data.rotType)
        self.assertEqual(visitInfo.observatory, data.observatory)
        self.assertEqual(visitInfo.instrumentLabel, data.instrumentLabel)
        self.assertEqual(visitInfo.weather, data.weather)
        self.assertEqual(visitInfo.localEra, localEra)
        self.assertEqual(visitInfo.boresightHourAngle, hourAngle)
        self.assertEqual(visitInfo.id, data.id)
        self.assertEqual(visitInfo.focusZ, data.focusZ)

    def testValueConstructor_data1(self):
        self._testValueConstructor(self.data1, self.localEra1, self.hourAngle1)
        self._testProperties(self.data1, self.localEra1, self.hourAngle1)

    def testValueConstructor_data2(self):
        self._testValueConstructor(self.data2, self.localEra2, self.hourAngle2)
        self._testProperties(self.data2, self.localEra2, self.hourAngle2)

    def testTablePersistence(self):
        """Test that VisitInfo can be round-tripped with current code.
        """
        for item in (self.data1, self.data2):
            tablePath = os.path.join(
                self.testDir, "testVisitInfo_testTablePersistence.fits")
            v1 = afwImage.VisitInfo(*item)
            v1.writeFits(tablePath)
            v2 = afwImage.VisitInfo.readFits(tablePath)
            self.assertEqual(v1, v2)
            os.unlink(tablePath)

    def _testFitsRead(self, data, filePath, version):
        """Test that old VersionInfo files are read correctly.

        Parameters
        ----------
        data : `VisitInfoData`
            The values expected to be stored in the file, or a
            superset thereof.
        filePath : `str`
            The file to test.
        version : `int`
            The VersionInfo persistence format used in ``filePath``.
        """
        visitInfo = afwImage.VisitInfo.readFits(filePath)

        if version >= 0:
            with self.assertWarns(FutureWarning):
                self.assertEqual(visitInfo.getExposureId(), data.exposureId)
            self.assertEqual(visitInfo.getExposureTime(), data.exposureTime)
            self.assertEqual(visitInfo.getDarkTime(), data.darkTime)
            self.assertEqual(visitInfo.getDate(), data.date)
            self.assertEqual(visitInfo.getUt1(), data.ut1)
            self.assertEqual(visitInfo.getEra(), data.era)
            self.assertEqual(visitInfo.getBoresightRaDec(), data.boresightRaDec)
            self.assertEqual(visitInfo.getBoresightAzAlt(), data.boresightAzAlt)
            self.assertEqual(visitInfo.getBoresightAirmass(),
                             data.boresightAirmass)
            self.assertEqual(visitInfo.getBoresightRotAngle(),
                             data.boresightRotAngle)
            self.assertEqual(visitInfo.getRotType(), data.rotType)
            self.assertEqual(visitInfo.getObservatory(), data.observatory)
            self.assertEqual(visitInfo.getWeather(), data.weather)
        if version >= 1:
            self.assertEqual(visitInfo.getInstrumentLabel(), data.instrumentLabel)
        else:
            self.assertEqual(visitInfo.getInstrumentLabel(), "")
        if version >= 2:
            self.assertEqual(visitInfo.getId(), data.id)
        else:
            self.assertEqual(visitInfo.getId(), 0)
        if version >= 3:
            self.assertEqual(visitInfo.getFocusZ(), data.focusZ)
        else:
            self.assertTrue(math.isnan(visitInfo.getFocusZ()))

    def testPersistenceVersions(self):
        """Test that older versions are handled appropriately.
        """
        dataDir = os.path.join(os.path.dirname(__file__), "data")

        # All files created by makeVisitInfo(self.data1).writeFits()
        self._testFitsRead(self.data1, os.path.join(dataDir, "visitInfo-noversion.fits"), 0)
        self._testFitsRead(self.data1, os.path.join(dataDir, "visitInfo-version-1.fits"), 1)
        self._testFitsRead(self.data1, os.path.join(dataDir, "visitInfo-version-2.fits"), 2)
        self._testFitsRead(self.data1, os.path.join(dataDir, "visitInfo-version-3.fits"), 3)

    def testSetVisitInfoMetadata(self):
        for item in (self.data1, self.data2):
            visitInfo = makeVisitInfo(item)
            metadata = PropertyList()
            afwImage.setVisitInfoMetadata(metadata, visitInfo)
            self.assertEqual(metadata.nameCount(), 23)
            self.assertEqual(metadata.getScalar("EXPID"), item.exposureId)
            self.assertEqual(metadata.getScalar("EXPTIME"), item.exposureTime)
            self.assertEqual(metadata.getScalar("DARKTIME"), item.darkTime)
            self.assertEqual(metadata.getScalar("DATE-AVG"),
                             item.date.toString(DateTime.TAI))
            self.assertEqual(metadata.getScalar("TIMESYS"), "TAI")
            self.assertEqual(metadata.getScalar("MJD-AVG-UT1"), item.ut1)
            self.assertEqual(metadata.getScalar("AVG-ERA"), item.era.asDegrees())
            self.assertEqual(metadata.getScalar("BORE-RA"),
                             item.boresightRaDec[0].asDegrees())
            self.assertEqual(metadata.getScalar("BORE-DEC"),
                             item.boresightRaDec[1].asDegrees())
            self.assertEqual(metadata.getScalar("BORE-AZ"),
                             item.boresightAzAlt[0].asDegrees())
            self.assertEqual(metadata.getScalar("BORE-ALT"),
                             item.boresightAzAlt[1].asDegrees())
            self.assertEqual(metadata.getScalar("BORE-AIRMASS"),
                             item.boresightAirmass)
            self.assertEqual(metadata.getScalar("BORE-ROTANG"),
                             item.boresightRotAngle.asDegrees())
            self.assertEqual(metadata.getScalar("ROTTYPE"),
                             RotTypeEnumNameDict[item.rotType])
            self.assertEqual(metadata.getScalar("OBS-LONG"),
                             item.observatory.getLongitude().asDegrees())
            self.assertEqual(metadata.getScalar("OBS-LAT"),
                             item.observatory.getLatitude().asDegrees())
            self.assertEqual(metadata.getScalar("OBS-ELEV"),
                             item.observatory.getElevation())
            self.assertEqual(metadata.getScalar("AIRTEMP"),
                             item.weather.getAirTemperature())
            self.assertEqual(metadata.getScalar("AIRPRESS"),
                             item.weather.getAirPressure())
            self.assertEqual(metadata.getScalar("HUMIDITY"),
                             item.weather.getHumidity())
            self.assertEqual(metadata.getScalar("INSTRUMENT"),
                             item.instrumentLabel)
            self.assertEqual(metadata.getScalar("IDNUM"),
                             item.id)
            self.assertEqual(metadata.getScalar("FOCUSZ"),
                             item.focusZ)

    def testSetVisitInfoMetadataMissingValues(self):
        """If a value is unknown then it should not be written to the metadata"""
        visitInfo = afwImage.VisitInfo()  # only rot type is known
        metadata = PropertyList()
        afwImage.setVisitInfoMetadata(metadata, visitInfo)
        self.assertEqual(metadata.getScalar("ROTTYPE"),
                         RotTypeEnumNameDict[afwImage.RotType.UNKNOWN])
        self.assertEqual(metadata.nameCount(), 1)

    def testStripVisitInfoKeywords(self):
        for argList in (self.data1, self.data2):
            visitInfo = afwImage.VisitInfo(*argList)
            metadata = PropertyList()
            afwImage.setVisitInfoMetadata(metadata, visitInfo)
            # add an extra keyword that will not be stripped
            metadata.set("EXTRA", 5)
            self.assertEqual(metadata.nameCount(), 24)
            afwImage.stripVisitInfoKeywords(metadata)
            self.assertEqual(metadata.nameCount(), 1)

    def _testIsEmpty(self, visitInfo):
        """Test that visitInfo is all NaN, 0, or empty string, as appropriate.
        """
        with self.assertWarns(FutureWarning):
            self.assertEqual(visitInfo.getExposureId(), 0)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))
        self.assertTrue(math.isnan(visitInfo.getDarkTime()))
        self.assertEqual(visitInfo.getDate(), DateTime())
        self.assertTrue(math.isnan(visitInfo.getUt1()))
        self.assertTrue(math.isnan(visitInfo.getEra().asDegrees()))
        for i in range(2):
            self.assertTrue(math.isnan(
                visitInfo.getBoresightRaDec()[i].asDegrees()))
            self.assertTrue(math.isnan(
                visitInfo.getBoresightAzAlt()[i].asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getBoresightAirmass()))
        self.assertTrue(math.isnan(
            visitInfo.getBoresightRotAngle().asDegrees()))
        self.assertEqual(visitInfo.getRotType(), afwImage.RotType.UNKNOWN)
        self.assertTrue(math.isnan(
            visitInfo.getObservatory().getLongitude().asDegrees()))
        self.assertTrue(math.isnan(
            visitInfo.getObservatory().getLatitude().asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getObservatory().getElevation()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getAirTemperature()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getAirPressure()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getHumidity()))
        self.assertTrue(math.isnan(visitInfo.getBoresightHourAngle()))
        self.assertEqual(visitInfo.getInstrumentLabel(), "")
        self.assertEqual(visitInfo.getId(), 0)
        self.assertTrue(math.isnan(visitInfo.getFocusZ()))

    def testEquals(self):
        """Test that identical VisitInfo objects compare equal, even if some fields are NaN.
        """
        # objects with "equal state" should be equal
        self.assertEqual(makeVisitInfo(self.data1), makeVisitInfo(self.data1))
        self.assertEqual(makeVisitInfo(self.data2), makeVisitInfo(self.data2))
        self.assertNotEqual(makeVisitInfo(self.data1), makeVisitInfo(self.data2))
        self.assertEqual(afwImage.VisitInfo(), afwImage.VisitInfo())

        # equality must be reflexive
        info = makeVisitInfo(self.data1)
        self.assertEqual(info, info)
        info = makeVisitInfo(self.data2)
        self.assertEqual(info, info)
        info = afwImage.VisitInfo()
        self.assertEqual(info, info)

        # commutativity and transitivity difficult to test with this setup

    def testMetadataConstructor(self):
        """Test the metadata constructor

        This constructor allows missing values
        """
        data = self.data1

        metadata = propertySetFromDict({})
        visitInfo = afwImage.VisitInfo(metadata)
        self._testIsEmpty(visitInfo)

        metadata = propertySetFromDict({"EXPID": data.exposureId})
        visitInfo = afwImage.VisitInfo(metadata)
        with self.assertWarns(FutureWarning):
            self.assertEqual(visitInfo.getExposureId(), data.exposureId)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))

        metadata = propertySetFromDict({"EXPTIME": data.exposureTime})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getExposureTime(), data.exposureTime)

        metadata = propertySetFromDict({"DARKTIME": data.darkTime})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDarkTime(), data.darkTime)

        metadata = propertySetFromDict(
            {"DATE-AVG": data.date.toString(DateTime.TAI), "TIMESYS": "TAI"})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDate(), data.date)

        # TIME-MID in UTC is an acceptable alternative to DATE-AVG
        metadata = propertySetFromDict(
            {"TIME-MID": data.date.toString(DateTime.UTC)})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDate(), data.date)

        # TIME-MID must be in UTC and TIMESYS is ignored
        metadata = propertySetFromDict({
            "TIME-MID": data.date.toString(DateTime.TAI) + "Z",
            "TIMESYS": "TAI",
        })
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertNotEqual(visitInfo.getDate(), data.date)

        # if both DATE-AVG and TIME-MID provided then use DATE-AVG
        # use the wrong time system for TIME-MID so if it is used, an error
        # will result
        metadata = propertySetFromDict({
            "DATE-AVG": data.date.toString(DateTime.TAI),
            "TIMESYS": "TAI",
            "TIME-MID": data.date.toString(DateTime.TAI) + "Z",
        })
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDate(), data.date)

        metadata = propertySetFromDict({"MJD-AVG-UT1": data.ut1})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getUt1(), data.ut1)

        metadata = propertySetFromDict({"AVG-ERA": data.era.asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getEra(), data.era)

        for i, key in enumerate(("BORE-RA", "BORE-DEC")):
            metadata = propertySetFromDict(
                {key: data.boresightRaDec[i].asDegrees()})
            visitInfo = afwImage.VisitInfo(metadata)
            self.assertEqual(visitInfo.getBoresightRaDec()
                             [i], data.boresightRaDec[i])

        for i, key in enumerate(("BORE-AZ", "BORE-ALT")):
            metadata = propertySetFromDict(
                {key: data.boresightAzAlt[i].asDegrees()})
            visitInfo = afwImage.VisitInfo(metadata)
            self.assertEqual(visitInfo.getBoresightAzAlt()
                             [i], data.boresightAzAlt[i])

        metadata = propertySetFromDict({"BORE-AIRMASS": data.boresightAirmass})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getBoresightAirmass(),
                         data.boresightAirmass)

        metadata = propertySetFromDict(
            {"BORE-ROTANG": data.boresightRotAngle.asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getBoresightRotAngle(),
                         data.boresightRotAngle)

        metadata = propertySetFromDict(
            {"ROTTYPE": RotTypeEnumNameDict[data.rotType]})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getRotType(), data.rotType)

        metadata = propertySetFromDict(
            {"OBS-LONG": data.observatory.getLongitude().asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getObservatory().getLongitude(),
                         data.observatory.getLongitude())

        metadata = propertySetFromDict(
            {"OBS-LAT": data.observatory.getLatitude().asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getObservatory().getLatitude(),
                         data.observatory.getLatitude())

        metadata = propertySetFromDict(
            {"OBS-ELEV": data.observatory.getElevation()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getObservatory().getElevation(),
                         data.observatory.getElevation())

        metadata = propertySetFromDict(
            {"AIRTEMP": data.weather.getAirTemperature()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getWeather().getAirTemperature(),
                         data.weather.getAirTemperature())

        metadata = propertySetFromDict(
            {"AIRPRESS": data.weather.getAirPressure()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getWeather().getAirPressure(),
                         data.weather.getAirPressure())

        metadata = propertySetFromDict(
            {"HUMIDITY": data.weather.getHumidity()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getWeather().getHumidity(),
                         data.weather.getHumidity())

        metadata = propertySetFromDict({"INSTRUMENT": data.instrumentLabel})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getInstrumentLabel(), data.instrumentLabel)

        metadata = propertySetFromDict({"IDNUM": data.id})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getId(), data.id)

        metadata = propertySetFromDict({"FOCUSZ": data.focusZ})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getFocusZ(), data.focusZ)

    def testConstructorKeywordArguments(self):
        """Test VisitInfo with named arguments"""
        data = self.data1

        visitInfo = afwImage.VisitInfo()
        self._testIsEmpty(visitInfo)

        visitInfo = afwImage.VisitInfo(exposureId=data.exposureId)
        with self.assertWarns(FutureWarning):
            self.assertEqual(visitInfo.getExposureId(), data.exposureId)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))

        visitInfo = afwImage.VisitInfo(exposureTime=data.exposureTime)
        self.assertEqual(visitInfo.getExposureTime(), data.exposureTime)

        visitInfo = afwImage.VisitInfo(darkTime=data.darkTime)
        self.assertEqual(visitInfo.getDarkTime(), data.darkTime)

        visitInfo = afwImage.VisitInfo(date=data.date)
        self.assertEqual(visitInfo.getDate(), data.date)

        visitInfo = afwImage.VisitInfo(ut1=data.ut1)
        self.assertEqual(visitInfo.getUt1(), data.ut1)

        visitInfo = afwImage.VisitInfo(era=data.era)
        self.assertEqual(visitInfo.getEra(), data.era)

        visitInfo = afwImage.VisitInfo(boresightRaDec=data.boresightRaDec)
        self.assertEqual(visitInfo.getBoresightRaDec(), data.boresightRaDec)

        visitInfo = afwImage.VisitInfo(boresightAzAlt=data.boresightAzAlt)
        self.assertEqual(visitInfo.getBoresightAzAlt(), data.boresightAzAlt)

        visitInfo = afwImage.VisitInfo(boresightAirmass=data.boresightAirmass)
        self.assertEqual(visitInfo.getBoresightAirmass(),
                         data.boresightAirmass)

        visitInfo = afwImage.VisitInfo(
            boresightRotAngle=data.boresightRotAngle)
        self.assertEqual(visitInfo.getBoresightRotAngle(),
                         data.boresightRotAngle)

        visitInfo = afwImage.VisitInfo(rotType=data.rotType)
        self.assertEqual(visitInfo.getRotType(), data.rotType)

        visitInfo = afwImage.VisitInfo(observatory=data.observatory)
        self.assertEqual(visitInfo.getObservatory(), data.observatory)

        visitInfo = afwImage.VisitInfo(weather=data.weather)
        self.assertEqual(visitInfo.getWeather(), data.weather)

        visitInfo = afwImage.VisitInfo(instrumentLabel=data.instrumentLabel)
        self.assertEqual(visitInfo.getInstrumentLabel(), data.instrumentLabel)

        visitInfo = afwImage.VisitInfo(id=data.id)
        self.assertEqual(visitInfo.getId(), data.id)

        visitInfo = afwImage.VisitInfo(focusZ=data.focusZ)
        self.assertEqual(visitInfo.getFocusZ(), data.focusZ)

    def testGoodRotTypes(self):
        """Test round trip of all valid rot types"""
        for rotType in RotTypeEnumNameDict:
            metadata = propertySetFromDict(
                {"ROTTYPE": RotTypeEnumNameDict[rotType]})
            visitInfo = afwImage.VisitInfo(metadata)
            self.assertEqual(visitInfo.getRotType(), rotType)

    def testBadRotTypes(self):
        """Test that invalid rot type names cannot be used to construct a VisitInfo"""
        for badRotTypeName in (
            "unknown",  # must be all uppercase
            "sky",  # must be all uppercase
            "Sky",  # must be all uppercase
            "SKY1",  # extra chars
            "HORIZONTAL",  # extra chars
        ):
            metadata = propertySetFromDict({"ROTTYPE": badRotTypeName})
            with self.assertRaises(lsst.pex.exceptions.RuntimeError):
                afwImage.VisitInfo(metadata)

    def test_str(self):
        """Check that we get something reasonable for str()"""
        visitInfo = makeVisitInfo(self.data1)
        string = str(visitInfo)
        self.assertIn("exposureId=10313423", string)
        self.assertIn("exposureTime=10.01", string)
        self.assertIn("darkTime=11.02", string)
        self.assertIn("rotType=1", string)

        # Check that it at least doesn't throw
        str(afwImage.VisitInfo())

    def testParallacticAngle(self):
        """Check that we get the same precomputed values for parallactic angle."""
        parallacticAngle = [141.39684140703142*degrees, 76.99982166973487*degrees]
        for item, parAngle in zip((self.data1, self.data2), parallacticAngle):
            visitInfo = afwImage.VisitInfo(era=item.era,
                                           boresightRaDec=item.boresightRaDec,
                                           observatory=item.observatory,
                                           )
            self.assertAnglesAlmostEqual(visitInfo.getBoresightParAngle(), parAngle)

    def testParallacticAngleNorthMeridian(self):
        """An observation on the Meridian that is North of zenith has a parallactic angle of pi radians."""
        meridianBoresightRA = self.data1.era + self.data1.observatory.getLongitude()
        northBoresightDec = self.data1.observatory.getLatitude() + 10.*degrees
        visitInfo = afwImage.VisitInfo(era=self.data1.era,
                                       boresightRaDec=SpherePoint(meridianBoresightRA,
                                                                  northBoresightDec),
                                       observatory=self.data1.observatory,
                                       )
        self.assertAnglesAlmostEqual(visitInfo.getBoresightParAngle(), Angle(np.pi))

    def testParallacticAngleSouthMeridian(self):
        """An observation on the Meridian that is South of zenith has a parallactic angle of zero."""
        meridianBoresightRA = self.data1.era + self.data1.observatory.getLongitude()
        southBoresightDec = self.data1.observatory.getLatitude() - 10.*degrees
        visitInfo = afwImage.VisitInfo(era=self.data1.era,
                                       boresightRaDec=SpherePoint(meridianBoresightRA,
                                                                  southBoresightDec),
                                       observatory=self.data1.observatory,
                                       )
        self.assertAnglesAlmostEqual(visitInfo.getBoresightParAngle(), Angle(0.))


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
