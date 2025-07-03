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

__all__ = ["CameraConfig", "DetectorConfig"]

import numpy as np
import lsst.pex.config as pexConfig
import lsst.geom as geom
from ._cameraGeom import Orientation
from ._transformConfig import TransformMapConfig


class DetectorConfig(pexConfig.Config):
    """A configuration that represents (and can be used to construct) a
    Detector.
    """
    transformDict = pexConfig.ConfigField(
        "Dictionary of camera transforms keyed on the transform type.", TransformMapConfig)
    name = pexConfig.Field("Name of detector slot", str)
    id = pexConfig.Field("ID of detector slot", int)
    bbox_x0 = pexConfig.Field("x0 of pixel bounding box", int)
    bbox_y0 = pexConfig.Field("y0 of pixel bounding box", int)
    bbox_x1 = pexConfig.Field("x1 of pixel bounding box", int)
    bbox_y1 = pexConfig.Field("y1 of pixel bounding box", int)
    detectorType = pexConfig.Field(
        "Detector type: SCIENCE=0, FOCUS=1, GUIDER=2, WAVEFRONT=3", int)
    physicalType = pexConfig.Field(
        "How this specific detector is constructed; e.g. CCD, E2V, HgCdTe ", str, default="CCD")
    serial = pexConfig.Field(
        "Serial string associated with this specific detector", str)
    offset_x = pexConfig.Field(
        "x offset from the origin of the camera in mm in the transposed system.", float)
    offset_y = pexConfig.Field(
        "y offset from the origin of the camera in mm in the transposed system.", float)
    offset_z = pexConfig.Field(
        "z offset from the origin of the camera in mm in the transposed system.", float, default=0.0)
    refpos_x = pexConfig.Field("x position of the reference point in the detector in pixels "
                               "in transposed coordinates.", float)
    refpos_y = pexConfig.Field("y position of the reference point in the detector in pixels "
                               "in transposed coordinates.", float)
    yawDeg = pexConfig.Field("yaw (rotation about z) of the detector in degrees. "
                             "This includes any necessary rotation to go from "
                             "detector coordinates to camera coordinates "
                             "after optional transposition.", float)
    pitchDeg = pexConfig.Field(
        "pitch (rotation about y) of the detector in degrees", float)
    rollDeg = pexConfig.Field(
        "roll (rotation about x) of the detector in degrees", float)
    pixelSize_x = pexConfig.Field("Pixel size in the x dimension in mm", float)
    pixelSize_y = pexConfig.Field("Pixel size in the y dimension in mm", float)

    # Depending on the choice of detector coordinates, the pixel grid may need
    # to be transposed before rotation to put it in camera coordinates.
    transposeDetector = pexConfig.Field(
        "Transpose the pixel grid before orienting in focal plane?", bool)

    crosstalk = pexConfig.ListField(
        dtype=float,
        doc=("Flattened crosstalk coefficient matrix; should have nAmps x nAmps entries. "
             "Once 'reshape'-ed, ``coeffs[i][j]`` is the fraction of the j-th amp present on the i-th amp."),
        optional=True
    )

    # Accessors to get "compiled" versions of parameters.
    def getCrosstalk(self, numAmps):
        """Return a 2-D numpy array of crosstalk coefficients of the proper shape"""
        if not self.crosstalk:
            return None

        if numAmps != int(np.sqrt(len(self.crosstalk))):
            numAmps = int(np.sqrt(len(self.crosstalk)))
        try:
            return np.array(self.crosstalk, dtype=np.float32).reshape((numAmps, numAmps))
        except Exception as e:
            raise RuntimeError(f"Cannot reshape 'crosstalk' coefficients to square matrix: {e}")

    @property
    def bbox(self):
        """Return the detector bounding box from the separate box endpoint
        values.
        """
        return geom.BoxI(geom.PointI(self.bbox_x0, self.bbox_y0),
                         geom.PointI(self.bbox_x1, self.bbox_y1))

    @property
    def offset(self):
        """Return the detector offset as a Point3D from the separate config
        values.
        """
        return geom.Point3D(self.offset_x, self.offset_y, self.offset_z)

    @property
    def refPos(self):
        """Return the detector reference position as a Point2D from the
        separate config values.
        """
        return geom.Point2D(self.refpos_x, self.refpos_y)

    @property
    def orientation(self):
        """Return the cameraGeom.Orientation() object defined by the
        configuration values.
        """
        return Orientation(self.offset, self.refPos,
                           geom.Angle(self.yawDeg, geom.degrees),
                           geom.Angle(self.pitchDeg, geom.degrees),
                           geom.Angle(self.rollDeg, geom.degrees))

    @property
    def pixelSize(self):
        """Return the pixel size as an Extent2D from the separate values.
        """
        return geom.Extent2D(self.pixelSize_x, self.pixelSize_y)


class CameraConfig(pexConfig.Config):
    """A configuration that represents (and can be used to construct) a Camera.
    """
    detectorList = pexConfig.ConfigDictField(
        "List of detector configs",
        keytype=int,
        itemtype=DetectorConfig,
    )
    transformDict = pexConfig.ConfigField(
        "Dictionary of camera transforms keyed on the transform type.",
        TransformMapConfig,
    )
    name = pexConfig.Field("Name of this camera", str)
    plateScale = pexConfig.Field("Plate scale of the camera in arcsec/mm", float)
    # Note that the radial transform will also apply a scaling, so all coefficients should be
    # scaled by the plate scale in appropriate units
    radialCoeffs = pexConfig.ListField("Coefficients for radial distortion", float)
    focalPlaneParity = pexConfig.Field(
        "Whether the FOCAL_PLANE <-> FIELD_ANGLE transform should flip the X axis.",
        dtype=bool,
        default=False,
    )
