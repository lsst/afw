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

import os, re
import eups

def ConfigureDependentProducts(productName, dependencyFilename="dependencies.dat"):
    """Process a product's dependency file, returning a list suitable for passing to SconsUtils.makeEnv"""
    productDir = eups.productDir(productName)
    if not productDir:
        raise RuntimeError, ("%s is not setup" % productName)

    dependencies = os.path.join(productDir, "etc", dependencyFilename)

    try:
        fd = open(dependencies)
    except:
        raise RuntimeError, ("Unable to lookup dependencies for %s" % productName)

    dependencies = []

    for line in fd.readlines():
        if re.search(r"^\s*#", line):
            continue

        mat = re.search(r"^(\S+):\s*$", line)
        if mat:
            dependencies += ConfigureDependentProducts(mat.group(1))
            continue
        #
        # Split the line into "" separated fields
        #
        line = re.sub(r"(^\s*|\s*,\s*|\s*$)", "", line) # remove whitespace and commas in the config file
        dependencies.append([f for f in re.split(r"['\"]", line) if f])

    return dependencies
    
    
