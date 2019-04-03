# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["makeCameraFromPath", "makeCameraFromAmpLists",
           "makeDetector", "copyDetector"]

import os.path
import lsst.geom
from lsst.afw.table import BaseCatalog
from .cameraGeomLib import FOCAL_PLANE, FIELD_ANGLE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS, CameraSys, \
    Detector, DetectorType, Orientation, TransformMap, Amplifier
from .camera import Camera
from .makePixelToTanPixel import makePixelToTanPixel
from .pupil import PupilFactory

cameraSysList = [FIELD_ANGLE, FOCAL_PLANE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS]
cameraSysMap = dict((sys.getSysName(), sys) for sys in cameraSysList)


def makeDetectorData(detectorConfig, amplifiers, focalPlaneToField):
    """Build a dictionary of Detector constructor keyword arguments.

    The returned dictionary can be passed as keyword arguments to the Detector
    constructor, providing all required arguments.  However, using these
    arguments directly constructs a Detector with knowledge of only the
    coordinate systems that are *directly* mapped to its own PIXELS coordinate
    system.  To construct Detectors with a shared TransformMap for the full
    Camera, use makeCameraFromCatalogs or makeCameraFromPath instead of
    calling this function or makeDetector directly.

    Parameters
    ----------
    detectorConfig : `lsst.pex.config.Config`
        Configuration for this detector.
    amplifiers : `list` of `~lsst.afw.cameraGeom.Amplifier`
        amplifier information for this detector
    focalPlaneToField : `lsst.afw.geom.TransformPoint2ToPoint2`
        FOCAL_PLANE to FIELD_ANGLE Transform

    Returns
    -------
    data : `dict`
        Contains the following keys: name, id, type, serial, bbox, orientation,
        pixelSize, transforms, amplifiers, and optionally crosstalk.
        The transforms key is a dictionary whose values are Transforms that map
        the Detector's PIXEL coordinate system to the CameraSys in the key.
    """

    data = dict(
        name=detectorConfig.name,
        id=detectorConfig.id,
        type=DetectorType(detectorConfig.detectorType),
        physicalType=detectorConfig.physicalType,
        serial=detectorConfig.serial,
        amplifiers=amplifiers,
        orientation=makeOrientation(detectorConfig),
        pixelSize=lsst.geom.Extent2D(detectorConfig.pixelSize_x, detectorConfig.pixelSize_y),
        bbox=lsst.geom.Box2I(
            minimum=lsst.geom.Point2I(detectorConfig.bbox_x0, detectorConfig.bbox_y0),
            maximum=lsst.geom.Point2I(detectorConfig.bbox_x1, detectorConfig.bbox_y1),
        ),
    )

    transforms = makeTransformDict(detectorConfig.transformDict.transforms)
    transforms[FOCAL_PLANE] = data["orientation"].makePixelFpTransform(data["pixelSize"])

    tanPixSys = CameraSys(TAN_PIXELS, detectorConfig.name)
    transforms[tanPixSys] = makePixelToTanPixel(
        bbox=data["bbox"],
        orientation=data["orientation"],
        focalPlaneToField=focalPlaneToField,
        pixelSizeMm=data["pixelSize"],
    )

    data["transforms"] = transforms

    crosstalk = detectorConfig.getCrosstalk(len(amplifiers))
    if crosstalk is not None:
        data["crosstalk"] = crosstalk

    return data


def makeDetector(detectorConfig, amplifiers, focalPlaneToField):
    """Make a Detector instance from a detector config and amp info catalog

    Parameters
    ----------
    detectorConfig : `lsst.pex.config.Config`
        Configuration for this detector.
    amplifiers : `list` of `~lsst.afw.cameraGeom.Amplifier`
        amplifier information for this detector
    focalPlaneToField : `lsst.afw.geom.TransformPoint2ToPoint2`
        FOCAL_PLANE to FIELD_ANGLE Transform

    Returns
    -------
    detector : `lsst.afw.cameraGeom.Detector`
        New Detector instance.
    """
    data = makeDetectorData(detectorConfig, amplifiers, focalPlaneToField)
    return Detector(**data)


def copyDetector(detector, amplifiers=None):
    """Return a copy of a Detector with possibly-updated amplifier information.

    No deep copies are made; the input transformDict is used unmodified

    Parameters
    ----------
    detector : `lsst.afw.cameraGeom.Detector`
        The Detector to clone
    amplifiers  The list of amplifiers to use; default use original

    Returns
    -------
    detector : `lsst.afw.cameraGeom.Detector`
        New Detector instance.
    """
    if amplifiers is None:
        amplifiers = detector.getAmplifiers()

    return Detector(detector.getName(), detector.getId(), detector.getType(),
                    detector.getSerial(), detector.getBBox(),
                    amplifiers, detector.getOrientation(), detector.getPixelSize(),
                    detector.getTransformMap(), detector.getCrosstalk(), detector.getPhysicalType())


def makeOrientation(detectorConfig):
    """Make an Orientation instance from a detector config

    Parameters
    ----------
    detectorConfig : `lsst.pex.config.Config`
        Configuration for this detector.

    Returns
    -------
    orientation : `lsst.afw.cameraGeom.Orientation`
        Location and rotation of the Detector.
    """
    offset = lsst.geom.Point2D(detectorConfig.offset_x, detectorConfig.offset_y)
    refPos = lsst.geom.Point2D(detectorConfig.refpos_x, detectorConfig.refpos_y)
    yaw = lsst.geom.Angle(detectorConfig.yawDeg, lsst.geom.degrees)
    pitch = lsst.geom.Angle(detectorConfig.pitchDeg, lsst.geom.degrees)
    roll = lsst.geom.Angle(detectorConfig.rollDeg, lsst.geom.degrees)
    return Orientation(offset, refPos, yaw, pitch, roll)


def makeTransformDict(transformConfigDict):
    """Make a dictionary of CameraSys: lsst.afw.geom.Transform from a config dict.

    Parameters
    ----------
    transformConfigDict : value obtained from a `lsst.pex.config.ConfigDictField`
        registry; keys are camera system names.

    Returns
    -------
    transforms : `dict`
        A dict of CameraSys or CameraSysPrefix: lsst.afw.geom.Transform
    """
    resMap = dict()
    if transformConfigDict is not None:
        for key in transformConfigDict:
            transform = transformConfigDict[key].transform.apply()
            resMap[CameraSys(key)] = transform
    return resMap


def makeCameraFromPath(cameraConfig, ampInfoPath, shortNameFunc,
                       pupilFactoryClass=PupilFactory):
    """Make a Camera instance from a directory of ampInfo files

    The directory must contain one ampInfo fits file for each detector in cameraConfig.detectorList.
    The name of each ampInfo file must be shortNameFunc(fullDetectorName) + ".fits".

    Parameters
    ----------
    cameraConfig : `CameraConfig`
        Config describing camera and its detectors.
    ampInfoPath : `str`
        Path to ampInfo data files.
    shortNameFunc : callable
        A function that converts a long detector name to a short one.
    pupilFactoryClass : `type`, optional
        Class to attach to camera; default is `lsst.afw.cameraGeom.PupilFactory`.

    Returns
    -------
    camera : `lsst.afw.cameraGeom.Camera`
        New Camera instance.
    """
    ampListDict = dict()
    for detectorConfig in cameraConfig.detectorList.values():
        shortName = shortNameFunc(detectorConfig.name)
        ampCatPath = os.path.join(ampInfoPath, shortName + ".fits")
        catalog = BaseCatalog.readFits(ampCatPath)
        ampListDict[detectorConfig.name] = [Amplifier.Builder.fromRecord(record).finish()
                                            for record in catalog]

    return makeCameraFromAmpLists(cameraConfig, ampListDict, pupilFactoryClass)


def makeCameraFromAmpLists(cameraConfig, ampListDict,
                           pupilFactoryClass=PupilFactory):
    """Construct a Camera instance from a dictionary of detector name: AmpInfoCatalog

    Parameters
    ----------
    cameraConfig : `CameraConfig`
        Config describing camera and its detectors.
    ampListDict : `dict`
        A dictionary of detector name: AmpInfoCatalog
    pupilFactoryClass : `type`, optional
        Class to attach to camera; `lsst.default afw.cameraGeom.PupilFactory`.

    Returns
    -------
    camera : `lsst.afw.cameraGeom.Camera`
        New Camera instance.
    """
    nativeSys = cameraSysMap[cameraConfig.transformDict.nativeSys]

    # nativeSys=FOCAL_PLANE seems to be assumed in various places in this file
    # (e.g. the definition of TAN_PIXELS), despite CameraConfig providing the
    # illusion that it's configurable.
    # Note that we can't actually get rid of the nativeSys config option
    # without breaking lots of on-disk camera configs.
    assert nativeSys == FOCAL_PLANE, "Cameras with nativeSys != FOCAL_PLANE are not supported."

    transformDict = makeTransformDict(cameraConfig.transformDict.transforms)
    focalPlaneToField = transformDict[FIELD_ANGLE]
    connections = [TransformMap.Connection(transform=transform, fromSys=nativeSys, toSys=toSys)
                   for toSys, transform in transformDict.items()]

    # First pass: build a list of all Detector ctor kwargs, minus the
    # transformMap (which needs information from all Detectors).
    detectorData = []
    for detectorConfig in cameraConfig.detectorList.values():

        # Get kwargs that could be used to construct each Detector
        # if we didn't care about giving each of them access to
        # all of the transforms.
        thisDetectorData = makeDetectorData(
            detectorConfig=detectorConfig,
            amplifiers=ampListDict[detectorConfig.name],
            focalPlaneToField=focalPlaneToField,
        )

        # Pull the transforms dictionary out of the data dict; we'll replace
        # it with a TransformMap argument later.
        thisDetectorTransforms = thisDetectorData.pop("transforms")

        # Save the rest of the Detector data dictionary for later
        detectorData.append(thisDetectorData)

        # For reasons I don't understand, some obs_ packages (e.g. HSC) set
        # nativeSys to None for their detectors (which doesn't seem to be
        # permitted by the config class!), but they really mean PIXELS. For
        # backwards compatibility we use that as the default...
        detectorNativeSysPrefix = cameraSysMap.get(detectorConfig.transformDict.nativeSys, PIXELS)

        # ...well, actually, it seems that we've always assumed down in C++
        # that the answer is always PIXELS without ever checking that it is.
        # So let's assert that it is, since there are hints all over this file
        # (e.g. the definition of TAN_PIXELS) that other parts of the codebase
        # have regularly made that assumption as well.  Note that we can't
        # actually get rid of the nativeSys config option without breaking
        # lots of on-disk camera configs.
        assert detectorNativeSysPrefix == PIXELS, "Detectors with nativeSys != PIXELS are not supported."
        detectorNativeSys = CameraSys(detectorNativeSysPrefix, detectorConfig.name)

        # Add this detector's transform dict to the connections list
        connections.extend(
            TransformMap.Connection(transform=transform, fromSys=detectorNativeSys, toSys=toSys)
            for toSys, transform in thisDetectorTransforms.items()
        )

    # Now that we've collected all of the Transforms, we can finally build the
    # (immutable) TransformMap.
    transformMap = TransformMap(nativeSys, connections)

    # Second pass through the detectorConfigs: actually make Detector instances
    detectorList = [Detector(transformMap=transformMap, **kw) for kw in detectorData]

    return Camera(cameraConfig.name, detectorList, transformMap, pupilFactoryClass)
