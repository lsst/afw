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
from __future__ import absolute_import, division, print_function
import math
import os
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.daf.base import DateTime, PropertySet, PropertyList
from lsst.afw.geom import degrees
from lsst.afw.coord import IcrsCoord, Coord, Observatory, Weather
import lsst.afw.image as afwImage

RotTypeEnumNameDict = {
    afwImage.RotType_UNKNOWN: "UNKNOWN",
    afwImage.RotType_SKY: "SKY",
    afwImage.RotType_HORIZON: "HORIZON",
    afwImage.RotType_MOUNT: "MOUNT",
}


def propertySetFromDict(keyValDict):
    """Make an lst.daf.base.PropertySet from a dict of key: value"""
    metadata = PropertySet()
    for key, val in keyValDict.items():
        metadata.set(key, val)
    return metadata


class VisitInfoTestCase(unittest.TestCase):
    """Test lsst.afw.image.VisitInfo, a simple struct-like class"""
    def setUp(self):
        self.testDir = os.path.dirname(__file__)

    def getArgTuples(self):
        """Return a collection of tuples of arbitrary values for constructing a VisitInfo

        Each argument tuple contains the following:
        - exposureId (int)
        - exposureTime (sec)
        - darkTime (sec)
        - date (lsst.daf.base.DateTime)
        - UT1 (MJD)
        - boresightRaDec (lsst.afw.coord.IcrsCoord)
        - boresightAzAlt (lsst.afw.coord.AzAltCoord)
        - airmass (float)
        - boresightRotAngle (lsst.afw.geom.Angle)
        - rotType (lsst.afw.image.RotType_x)
        - observatory (lsst.afw.coord.Observatory)
        - weather (lsst.afw.coord.Weather)
        """
        return (
            (
                10313423,
                10.01,
                11.02,
                DateTime(65321.1, DateTime.MJD, DateTime.TAI),
                12345.1,
                45.1*degrees,
                IcrsCoord(23.1*degrees, 73.2*degrees),
                Coord(134.5*degrees, 33.3*degrees),
                1.73,
                73.2*degrees,
                afwImage.RotType_SKY,
                Observatory(11.1*degrees, 22.2*degrees, 0.333),
                Weather(1.1, 2.2, 34.5),
            ),
            (
                1,
                15.5,
                17.8,
                DateTime(55321.2, DateTime.MJD, DateTime.TAI),
                312345.1,
                25.1*degrees,
                IcrsCoord(2.1*degrees, 33.2*degrees),
                Coord(13.5*degrees, 83.3*degrees),
                2.05,
                -53.2*degrees,
                afwImage.RotType_HORIZON,
                Observatory(22.2*degrees, 33.3*degrees, 0.444),
                Weather(2.2, 3.3, 44.4),
            ),
        )

    def testValueConstructor(self):
        for (
            exposureId,
            exposureTime,
            darkTime,
            date,
            ut1,
            era,
            boresightRaDec,
            boresightAzAlt,
            boresightAirmass,
            boresightRotAngle,
            rotType,
            observatory,
            weather,
        ) in self.getArgTuples():
            visitInfo = afwImage.VisitInfo(
                exposureId,
                exposureTime,
                darkTime,
                date,
                ut1,
                era,
                boresightRaDec,
                boresightAzAlt,
                boresightAirmass,
                boresightRotAngle,
                rotType,
                observatory,
                weather,
            )
            self.assertEqual(visitInfo.getExposureId(), exposureId)
            self.assertEqual(visitInfo.getExposureTime(), exposureTime)
            self.assertEqual(visitInfo.getDarkTime(), darkTime)
            self.assertEqual(visitInfo.getDate(), date)
            self.assertEqual(visitInfo.getUt1(), ut1)
            self.assertEqual(visitInfo.getEra(), era)
            self.assertEqual(visitInfo.getBoresightRaDec(), boresightRaDec)
            self.assertEqual(visitInfo.getBoresightAzAlt(), boresightAzAlt)
            self.assertEqual(visitInfo.getBoresightAirmass(), boresightAirmass)
            self.assertEqual(visitInfo.getBoresightRotAngle(), boresightRotAngle)
            self.assertEqual(visitInfo.getRotType(), rotType)
            self.assertEqual(visitInfo.getObservatory(), observatory)
            self.assertEqual(visitInfo.getWeather(), weather)

    def testTablePersistence(self):
        for valueList in self.getArgTuples():
            tablePath = os.path.join(self.testDir, "testVisitInfo_testTablePersistence.fits")
            v1 = afwImage.VisitInfo(*valueList)
            v1.writeFits(tablePath)
            v2 = afwImage.VisitInfo.readFits(tablePath)
            self.assertEqual(v1, v2)
            os.unlink(tablePath)

    def testSetVisitInfoMetadata(self):
        for (
            exposureId,
            exposureTime,
            darkTime,
            date,
            ut1,
            era,
            boresightRaDec,
            boresightAzAlt,
            boresightAirmass,
            boresightRotAngle,
            rotType,
            observatory,
            weather,
        ) in self.getArgTuples():
            visitInfo = afwImage.VisitInfo(
                exposureId,
                exposureTime,
                darkTime,
                date,
                ut1,
                era,
                boresightRaDec,
                boresightAzAlt,
                boresightAirmass,
                boresightRotAngle,
                rotType,
                observatory,
                weather,
            )
            metadata = PropertyList()
            afwImage.setVisitInfoMetadata(metadata, visitInfo)
            self.assertEqual(metadata.nameCount(), 20)
            self.assertEqual(metadata.get("EXPID"), exposureId)
            self.assertEqual(metadata.get("EXPTIME"), exposureTime)
            self.assertEqual(metadata.get("DARKTIME"), darkTime)
            self.assertEqual(metadata.get("DATE-AVG"), date.toString(DateTime.TAI))
            self.assertEqual(metadata.get("TIMESYS"), "TAI")
            self.assertEqual(metadata.get("MJD-AVG-UT1"), ut1)
            self.assertEqual(metadata.get("AVG-ERA"), era.asDegrees())
            self.assertEqual(metadata.get("BORE-RA"), boresightRaDec[0].asDegrees())
            self.assertEqual(metadata.get("BORE-DEC"), boresightRaDec[1].asDegrees())
            self.assertEqual(metadata.get("BORE-AZ"), boresightAzAlt[0].asDegrees())
            self.assertEqual(metadata.get("BORE-ALT"), boresightAzAlt[1].asDegrees())
            self.assertEqual(metadata.get("BORE-AIRMASS"), boresightAirmass)
            self.assertEqual(metadata.get("BORE-ROTANG"), boresightRotAngle.asDegrees())
            self.assertEqual(metadata.get("ROTTYPE"), RotTypeEnumNameDict[rotType])
            self.assertEqual(metadata.get("OBS-LONG"), observatory.getLongitude().asDegrees())
            self.assertEqual(metadata.get("OBS-LAT"), observatory.getLatitude().asDegrees())
            self.assertEqual(metadata.get("OBS-ELEV"), observatory.getElevation())
            self.assertEqual(metadata.get("AIRTEMP"), weather.getAirTemperature())
            self.assertEqual(metadata.get("AIRPRESS"), weather.getAirPressure())
            self.assertEqual(metadata.get("HUMIDITY"), weather.getHumidity())

    def testSetVisitInfoMetadataMissingValues(self):
        """If a value is unknown then it should not be written to the metadata"""
        visitInfo = afwImage.makeVisitInfo()  # only rot type is known
        metadata = PropertyList()
        afwImage.setVisitInfoMetadata(metadata, visitInfo)
        self.assertEqual(metadata.get("ROTTYPE"), RotTypeEnumNameDict[afwImage.RotType_UNKNOWN])
        self.assertEqual(metadata.nameCount(), 1)

    def testStripVisitInfoKeywords(self):
        for argList in self.getArgTuples():
            visitInfo = afwImage.VisitInfo(*argList)
            metadata = PropertyList()
            afwImage.setVisitInfoMetadata(metadata, visitInfo)
            metadata.set("EXTRA", 5)  # add an extra keyword that will not be stripped
            self.assertEqual(metadata.nameCount(), 21)
            afwImage.stripVisitInfoKeywords(metadata)
            self.assertEqual(metadata.nameCount(), 1)

    def testMetadataConstructor(self):
        """Test the metadata constructor

        This constructor allows missing values
        """
        (
            exposureId,
            exposureTime,
            darkTime,
            date,
            ut1,
            era,
            boresightRaDec,
            boresightAzAlt,
            boresightAirmass,
            boresightRotAngle,
            rotType,
            observatory,
            weather,
        ) = self.getArgTuples()[0]

        metadata = propertySetFromDict({})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getExposureId(), 0)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))
        self.assertTrue(math.isnan(visitInfo.getDarkTime()))
        self.assertEqual(visitInfo.getDate(), DateTime())
        self.assertTrue(math.isnan(visitInfo.getUt1()))
        self.assertTrue(math.isnan(visitInfo.getEra().asDegrees()))
        for i in range(2):
            self.assertTrue(math.isnan(visitInfo.getBoresightRaDec()[i].asDegrees()))
            self.assertTrue(math.isnan(visitInfo.getBoresightAzAlt()[i].asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getBoresightAirmass()))
        self.assertTrue(math.isnan(visitInfo.getBoresightRotAngle().asDegrees()))
        self.assertEqual(visitInfo.getRotType(), afwImage.RotType_UNKNOWN)
        self.assertTrue(math.isnan(visitInfo.getObservatory().getLongitude().asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getObservatory().getLatitude().asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getObservatory().getElevation()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getAirTemperature()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getAirPressure()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getHumidity()))

        metadata = propertySetFromDict({"EXPID": exposureId})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getExposureId(), exposureId)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))

        metadata = propertySetFromDict({"EXPTIME": exposureTime})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getExposureTime(), exposureTime)

        metadata = propertySetFromDict({"DARKTIME": darkTime})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDarkTime(), darkTime)

        metadata = propertySetFromDict({"DATE-AVG": date.toString(DateTime.TAI), "TIMESYS": "TAI"})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDate(), date)

        # TIME-MID in UTC is an acceptable alternative to DATE-AVG
        metadata = propertySetFromDict({"TIME-MID": date.toString(DateTime.UTC)})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDate(), date)

        # TIME-MID must be in UTC and TIMESYS is ignored
        metadata = propertySetFromDict({
            "TIME-MID": date.toString(DateTime.TAI) + "Z",
            "TIMESYS": "TAI",
        })
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertNotEqual(visitInfo.getDate(), date)

        # if both DATE-AVG and TIME-MID provided then use DATE-AVG
        # use the wrong time system for TIME-MID so if it is used, an error will result
        metadata = propertySetFromDict({
            "DATE-AVG": date.toString(DateTime.TAI),
            "TIMESYS": "TAI",
            "TIME-MID": date.toString(DateTime.TAI) + "Z",
        })
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getDate(), date)

        metadata = propertySetFromDict({"MJD-AVG-UT1": ut1})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getUt1(), ut1)

        metadata = propertySetFromDict({"AVG-ERA": era.asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getEra(), era)

        for i, key in enumerate(("BORE-RA", "BORE-DEC")):
            metadata = propertySetFromDict({key: boresightRaDec[i].asDegrees()})
            visitInfo = afwImage.VisitInfo(metadata)
            self.assertEqual(visitInfo.getBoresightRaDec()[i], boresightRaDec[i])

        for i, key in enumerate(("BORE-AZ", "BORE-ALT")):
            metadata = propertySetFromDict({key: boresightAzAlt[i].asDegrees()})
            visitInfo = afwImage.VisitInfo(metadata)
            self.assertEqual(visitInfo.getBoresightAzAlt()[i], boresightAzAlt[i])

        metadata = propertySetFromDict({"BORE-AIRMASS": boresightAirmass})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getBoresightAirmass(), boresightAirmass)

        metadata = propertySetFromDict({"BORE-ROTANG": boresightRotAngle.asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getBoresightRotAngle(), boresightRotAngle)

        metadata = propertySetFromDict({"ROTTYPE": RotTypeEnumNameDict[rotType]})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getRotType(), rotType)

        metadata = propertySetFromDict({"OBS-LONG": observatory.getLongitude().asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getObservatory().getLongitude(), observatory.getLongitude())

        metadata = propertySetFromDict({"OBS-LAT": observatory.getLatitude().asDegrees()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getObservatory().getLatitude(), observatory.getLatitude())

        metadata = propertySetFromDict({"OBS-ELEV": observatory.getElevation()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getObservatory().getElevation(), observatory.getElevation())

        metadata = propertySetFromDict({"AIRTEMP": weather.getAirTemperature()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getWeather().getAirTemperature(), weather.getAirTemperature())

        metadata = propertySetFromDict({"AIRPRESS": weather.getAirPressure()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getWeather().getAirPressure(), weather.getAirPressure())

        metadata = propertySetFromDict({"HUMIDITY": weather.getHumidity()})
        visitInfo = afwImage.VisitInfo(metadata)
        self.assertEqual(visitInfo.getWeather().getHumidity(), weather.getHumidity())

    def testMakeVisitInfo(self):
        """Test the makeVisitInfo factory function"""
        (
            exposureId,
            exposureTime,
            darkTime,
            date,
            ut1,
            era,
            boresightRaDec,
            boresightAzAlt,
            boresightAirmass,
            boresightRotAngle,
            rotType,
            observatory,
            weather,
        ) = self.getArgTuples()[0]

        visitInfo = afwImage.makeVisitInfo()
        self.assertEqual(visitInfo.getExposureId(), 0)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))
        self.assertTrue(math.isnan(visitInfo.getDarkTime()))
        self.assertEqual(visitInfo.getDate(), DateTime())
        self.assertTrue(math.isnan(visitInfo.getUt1()))
        self.assertTrue(math.isnan(visitInfo.getEra().asDegrees()))
        for i in range(2):
            self.assertTrue(math.isnan(visitInfo.getBoresightRaDec()[i].asDegrees()))
            self.assertTrue(math.isnan(visitInfo.getBoresightAzAlt()[i].asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getBoresightAirmass()))
        self.assertTrue(math.isnan(visitInfo.getBoresightRotAngle().asDegrees()))
        self.assertEqual(visitInfo.getRotType(), afwImage.RotType_UNKNOWN)
        self.assertTrue(math.isnan(visitInfo.getObservatory().getLongitude().asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getObservatory().getLatitude().asDegrees()))
        self.assertTrue(math.isnan(visitInfo.getObservatory().getElevation()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getAirTemperature()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getAirPressure()))
        self.assertTrue(math.isnan(visitInfo.getWeather().getHumidity()))

        visitInfo = afwImage.makeVisitInfo(exposureId=exposureId)
        self.assertEqual(visitInfo.getExposureId(), exposureId)
        self.assertTrue(math.isnan(visitInfo.getExposureTime()))

        visitInfo = afwImage.makeVisitInfo(exposureTime=exposureTime)
        self.assertEqual(visitInfo.getExposureTime(), exposureTime)

        visitInfo = afwImage.makeVisitInfo(darkTime=darkTime)
        self.assertEqual(visitInfo.getDarkTime(), darkTime)

        visitInfo = afwImage.makeVisitInfo(date=date)
        self.assertEqual(visitInfo.getDate(), date)

        visitInfo = afwImage.makeVisitInfo(ut1=ut1)
        self.assertEqual(visitInfo.getUt1(), ut1)

        visitInfo = afwImage.makeVisitInfo(era=era)
        self.assertEqual(visitInfo.getEra(), era)

        visitInfo = afwImage.makeVisitInfo(boresightRaDec=boresightRaDec)
        self.assertEqual(visitInfo.getBoresightRaDec(), boresightRaDec)

        visitInfo = afwImage.makeVisitInfo(boresightAzAlt=boresightAzAlt)
        self.assertEqual(visitInfo.getBoresightAzAlt(), boresightAzAlt)

        visitInfo = afwImage.makeVisitInfo(boresightAirmass=boresightAirmass)
        self.assertEqual(visitInfo.getBoresightAirmass(), boresightAirmass)

        visitInfo = afwImage.makeVisitInfo(boresightRotAngle=boresightRotAngle)
        self.assertEqual(visitInfo.getBoresightRotAngle(), boresightRotAngle)

        visitInfo = afwImage.makeVisitInfo(rotType=rotType)
        self.assertEqual(visitInfo.getRotType(), rotType)

        visitInfo = afwImage.makeVisitInfo(observatory=observatory)
        self.assertEqual(visitInfo.getObservatory(), observatory)

        visitInfo = afwImage.makeVisitInfo(weather=weather)
        self.assertEqual(visitInfo.getWeather(), weather)

    def testGoodRotTypes(self):
        """Test round trip of all valid rot types"""
        for rotType in RotTypeEnumNameDict:
            metadata = propertySetFromDict({"ROTTYPE": RotTypeEnumNameDict[rotType]})
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


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
