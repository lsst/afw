from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2008-2016 LSST Corporation.
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

import re
import lsst.pex.policy as pexPolicy
from lsst.afw.cameraGeom import TAN_PIXELS
import lsst.afw.detection as afwDetect
from . import imageLib as afwImage

def clipImage(im, minClip, maxClip):
    """Clip an image to lie between minClip and maxclip (None to ignore)"""

    if re.search("::MaskedImage<", im.__repr__()):
        mi = im
    else:
        mi = afwImage.makeMaskedImage(im, afwImage.MaskU(im.getDimensions()))

    if minClip is not None:
        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(-minClip, afwDetect.Threshold.VALUE, False))
        afwDetect.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), minClip)

    if maxClip is not None:
        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(maxClip))
        afwDetect.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), maxClip)

def getDistortedWcs(exposureInfo, log=None):
        """!Get a WCS from an exposureInfo, with distortion terms if possible

        If the WCS in the exposure is a pure TAN WCS and distortion information is available
        in the exposure's Detector, then return a DistortedTanWcs that combines the
        distortion information with the pure TAN WCS.
        Otherwise return the WCS in the exposureInfo without modification.

        This function is intended as a temporary workaround until ISR puts a WCS with distortion information
        into its exposures.

        @param[in] exposureInfo  exposure information (an lsst.afw.image.ExposureInfo),
            e.g. from exposure.getInfo()
        @param[in] log  an lsst.pex.logging.Log or None; if specified then a warning is logged if:
            - the exposureInfo's WCS has no distortion and cannot be cast to a TanWcs
            - the expousureInfo's detector has no TAN_PIXELS transform (distortion information)
        @throw RuntimeError if exposureInfo has no WCS.
        """
        if not exposureInfo.hasWcs():
            raise RuntimeError("exposure must have a WCS")
        wcs = exposureInfo.getWcs()
        if not wcs.hasDistortion() and exposureInfo.hasDetector():
            # warn but continue if TAN_PIXELS not present or the initial WCS is not a TanWcs;
            # other errors indicate a bug that should raise an exception
            detector = exposureInfo.getDetector()
            try:
                pixelsToTanPixels = detector.getTransform(TAN_PIXELS)
                tanWcs = afwImage.TanWcs.cast(wcs)
            except Exception as e:
                if log:
                    log.warn("Could not create a DistortedTanWcs: %s" % (e,))
            else:
                wcs = afwImage.DistortedTanWcs(tanWcs, pixelsToTanPixels)
        return wcs

def resetFilters():
    """Reset registry of filters and filter properties"""
    afwImage.Filter.reset()
    afwImage.FilterProperty.reset()

def defineFilter(name, lambdaEff, alias=[], force=False):
    """Define a filter and its properties in the filter registry"""
    prop = afwImage.FilterProperty(name, lambdaEff, force)
    afwImage.Filter.define(prop)
    if isinstance(alias, basestring):
        afwImage.Filter.defineAlias(name, alias)
    else:
        for a in alias:
            afwImage.Filter.defineAlias(name, a)

def defineFiltersFromPolicy(filterPolicy, reset=False):
    """Process a Policy and define the filters"""

    if reset:
        afwImage.Filter.reset()
        afwImage.FilterProperty.reset()
    #
    # Process the Policy and define the filters
    #
    policyFile = pexPolicy.DefaultPolicyFile("afw", "FilterDictionary.paf", "policy")
    defPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

    filterPolicy.mergeDefaults(defPolicy.getDictionary())

    for p in filterPolicy.getArray("Filter"):
        afwImage.Filter.define(afwImage.FilterProperty(p.get("name"), p))
        if p.exists("alias"):
            for a in p.getArray("alias"):
                afwImage.Filter.defineAlias(p.get("name"), a)

class CalibNoThrow(object):
    """A class intended to be used with python's with statement, to return NaNs for negative fluxes
    instead of raising exceptions (exceptions may be raised for other purposes).

E.g.
     with CalibNoThrow():
         ax.plot([exposure.getCalib().getMagnitude(a) for a in candAmps], zGood[:,k], 'b+')
    """
    def __enter__(self):
        self._throwOnNegative = afwImage.Calib.getThrowOnNegativeFlux()
        afwImage.Calib.setThrowOnNegativeFlux(False)

    def __exit__(self, *args):
        afwImage.Calib.setThrowOnNegativeFlux(self._throwOnNegative)
