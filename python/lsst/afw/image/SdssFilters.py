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

from lsst.afw.image.utils import defineFilter

# Effective wavelengths from Fukugita et al., 1996AJ....111.1748F
# Table 2a (1.2 airmass, first row)
defineFilter("u'", 355.7)
defineFilter("g'", 482.5)
defineFilter("r'", 626.1)
defineFilter("i'", 767.2)
defineFilter("z'", 909.7)
