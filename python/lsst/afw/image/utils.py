# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import lsst.afw.detection as detection
from . import imageLib as afwImage
import numpy

def clipImage(im, minClip, maxClip):
    """Clip an image to lie between minClip and maxclip (None to ignore)"""

    if re.search("::MaskedImage<", im.__repr__()):
        mi = im
    else:
        mi = afwImage.makeMaskedImage(im, afwImage.MaskU(im.getDimensions()))

    if minClip is not None:
        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(-minClip, afwDetect.Threshold.VALUE, False))
        afwDetect.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), minClip)

    if maxclip is not None:
        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(maxclip))
        afwDetect.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), maxclip)

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
