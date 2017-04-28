from __future__ import absolute_import, division, print_function

from lsst.afw.image.utils import defineFilter

# Effective wavelengths from Fukugita et al., 1996AJ....111.1748F
# Table 2a (1.2 airmass, first row)
defineFilter("u'", 355.7)
defineFilter("g'", 482.5)
defineFilter("r'", 626.1)
defineFilter("i'", 767.2)
defineFilter("z'", 909.7)
