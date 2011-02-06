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

"""lsst.afw.geom.ellipses
"""
from ellipsesLib import (
    LogShear,
    Distortion,
    GeometricRadius,
    ArithmeticRadius,
    LogGeometricRadius,
    LogArithmeticRadius,
    BaseCore,
    Axes,
    Quadrupole,
    SeparableDistortionGeometricRadius,
    SeparableDistortionArithmeticRadius,
    SeparableDistortionLogGeometricRadius,
    SeparableDistortionLogArithmeticRadius,
    SeparableLogShearGeometricRadius,
    SeparableLogShearArithmeticRadius,
    SeparableLogShearLogGeometricRadius,
    SeparableLogShearLogArithmeticRadius,
    Ellipse,
    Parametric
)

Separable = {
    (Distortion, GeometricRadius):SeparableDistortionGeometricRadius,
    (Distortion, ArithmeticRadius):SeparableDistortionArithmeticRadius,
    (Distortion, LogGeometricRadius):SeparableDistortionLogGeometricRadius,
    (Distortion, LogArithmeticRadius):SeparableDistortionLogArithmeticRadius,
    (LogShear, GeometricRadius):SeparableLogShearGeometricRadius,
    (LogShear, ArithmeticRadius):SeparableLogShearArithmeticRadius,
    (LogShear, LogGeometricRadius):SeparableLogShearLogGeometricRadius,
    (LogShear, LogArithmeticRadius):SeparableLogShearLogArithmeticRadius
}

