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

import lsst.pex.config as pexConfig

class FilterConfig(pexConfig.Config):
    name = pexConfig.Field(dtype=str, doc="Filter name")
    alias = pexConfig.ListField(dtype=str, doc="List of alternative names", optional=True)
    lambdaEff = pexConfig.Field(dtype=float, doc="Effective wavelength", optional=True)

class FilterSetConfig(pexConfig.Config):
    description = pexConfig.Field(dtype=str, doc="Description of filter set")
    filters = pexConfig.ListField(dtype=FilterConfig, doc="List of FilterConfig")

