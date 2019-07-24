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

__all__ = ['TransformMapConfig', 'transformDictFromYaml']

import astshim as ast
import numpy as np

import lsst.pex.config as pexConfig
import lsst.geom as geom
import lsst.afw.geom as afwGeom
from .cameraGeomLib import CameraSys, FOCAL_PLANE  # , FIELD_ANGLE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS


class TransformMapConfig(pexConfig.Config):
    transforms = pexConfig.ConfigDictField(
        doc="Dict of coordinate system name: TransformConfig",
        keytype=str,
        itemtype=afwGeom.TransformConfig,
    )
    nativeSys = pexConfig.Field(
        doc="Name of reference coordinate system",
        dtype=str,
        optional=False,
    )

    def transformsToDict(self):
        """Make a dictionary of CameraSys: lsst.afw.geom.Transform from a config dict.

        Returns
        -------
        transforms : `dict`
           A dict of CameraSys or CameraSysPrefix: lsst.afw.geom.Transform
        """
        resMap = dict()
        for key in self.transforms:
            transform = self.transforms[key].transform.apply()
            resMap[CameraSys(key)] = transform
        return resMap


def transformDictFromYaml(plateScale, paramDict):
    """Make a dictionary of TransformPoint2Point2s from yaml, mapping from nativeSys.

    Parameters
    ----------
    plateScale : `lsst.geom.Angle`
        The size of a pixel in angular units/mm (e.g. 20 arcsec/mm for LSST).
    paramDict : `dict`
        A `dict` specifying parameters of transforms.  Keys are the camera system names.

    Returns
    -------
    transforms : `dict`
        A dict of `lsst.afw.cameraGeom.CameraSys` : `lsst.afw.geom.TransformPoint2ToPoint2`

    The result dict's keys are `~lsst.afw.cameraGeom.CameraSys`, and
    the values are Transforms *from* ``nativeSys`` to `CameraSys`.

    The ``nativeSys`` is defined to be `FOCAL_PLANE`, by convention, and as cameras without this
    nativeSys are defined as unsupported.
    """
    nativeSys = FOCAL_PLANE
    resMap = dict()

    knownTransformTypes = ['affine', 'radial', 'hsc', 'poly']
    inDict = paramDict
    if 'transforms' in inDict:
        inDict = inDict['transforms']

    for key, transform in inDict.items():
        transformType = transform.get('transformType', None)
        if transformType is None:
            try:
                tForm = transform['transform']['values']['hsc']
                transform = tForm
                transformType = 'hsc'
            except:
                print("This is a warning.")
        if transformType not in knownTransformTypes:
            raise RuntimeError("Saw unknown transform type for %s: %s (known types are: [%s])" % (
                key, transform['transformType'], ', '.join(knownTransformTypes)))

        if transformType == 'affine':
            affine = geom.AffineTransform(np.array(transform['linear']),
                                          np.array(transform['translation']))
            transform = afwGeom.makeTransform(affine)

        elif transformType == 'radial':
            # radial coefficients of the form [0, 1 (no units), C2 (rad), usually 0, C3 (rad^2), ...]
            # Radial distortion is modeled as a radial polynomial that converts from focal plane radius
            # (in mm) to field angle (in radians). The provided coefficients are divided by the plate
            # scale (in radians/mm) meaning that C1 is always 1.
            radialCoeffs = np.array(transform['coeffs'])

            radialCoeffs *= plateScale.asRadians()
            transform = afwGeom.makeRadialTransform(radialCoeffs)

        elif transformType == 'hsc' or transformType == 'poly':
            forwardCoeffs = makeAstPolyMapCoeffs(transform['ccdToSkyOrder'],
                                                 transform['xCcdToSky'],
                                                 transform['yCcdToSky'])

            # Note that the actual error can be somewhat larger than TolInverse;
            # the max error seen has been less than 2, so scale conservatively.
            ccdToSky = ast.PolyMap(forwardCoeffs, 2, "IterInverse=1, TolInverse=%s, NIterInverse=%s" %
                                   (transform['tolerance'] / 2.0, transform['maxIter']))
            plateScale = transform['plateScale']
            if type(plateScale) is not geom.Angle:
                plateScale = geom.Angle(plateScale, geom.arcseconds)
            fullMapping = ccdToSky.then(ast.ZoomMap(2, plateScale.asRadians()))
            transform = afwGeom.TransformPoint2ToPoint2(fullMapping)

        else:
            raise RuntimeError("Impossible condition \"%s\" is not in [%s])" % (
                transform['transformType'], ", ".join(knownTransformTypes)))

        resMap[CameraSys(key)] = transform
    print(nativeSys)
    return resMap


def makeAstPolyMapCoeffs(order, xCoeffs, yCoeffs):
    """Convert polynomial coefficients in HSC format to AST PolyMap format

    Paramaters
    ----------
    order: `int`
        Polynomial order
    xCoeffs, yCoeffs: `list` of `float`
        Forward or inverse polynomial coefficients for the x and y axes
        of output, in this order:
            x0y0, x0y1, ...x0yN, x1y0, x1y1, ...x1yN-1, ...
        where N is the polynomial order.

    Returns
    -------
    Forward or inverse coefficients for `astshim.PolyMap`
    as a 2-d numpy array.
    """
    nCoeffs = (order + 1) * (order + 2) // 2
    if len(xCoeffs) != nCoeffs:
        raise ValueError("found %s xCcdToSky params; need %s" % (len(xCoeffs), nCoeffs))
    if len(yCoeffs) != nCoeffs:
        raise ValueError("found %s yCcdToSky params; need %s" % (len(yCoeffs), nCoeffs))

    coeffs = np.zeros([nCoeffs * 2, 4])
    i = 0
    for nx in range(order + 1):
        for ny in range(order + 1 - nx):
            coeffs[i] = [xCoeffs[i], 1, nx, ny]
            coeffs[i + nCoeffs] = [yCoeffs[i], 2, nx, ny]
            i += 1
    assert i == nCoeffs
    return coeffs
