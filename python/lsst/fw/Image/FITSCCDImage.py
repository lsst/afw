#####################################################
#
# A CCDImage is a subtype of Image restricted to a single HDU
#
#####################################################

__all__ = ["FITSCCDImage"]

import os
import math
import pyfits
from lsst.fw.Policy import Policy
from lsst.support.PySextractor import *
from lsst.fw.Collection import *
from RO.StringUtil import *

from Image import Image
from CCDGeom import CCDGeom
from SkyRegion import SkyRegion
from ImageMetaData import ImageMetaData

class FITSCCDImage(Image):
    """
    CCDImage is a python realization of the LSST UML class CCDImage
    """
    #---------------------------------------------------
    def __init__(self, arg, policyFile=None, **kws):
        """
        Construct a CCDImage from either a FITS file name or a single HDU

        Input
            arg             Arguments passed directly to Image initialization
                            Format:     ; Default: none
            policyFile      Filename for class Policy specification;
                            Format: string
                            Default:
                                if none provided, use "MosaicImage.conf"
            kws             Keyword dictionary which is used, along with
                            Policy, to manage processing; Format: dictionary
        Return
            none
        Raise
            ValueError      if no filename and no HDU argument provided
            ValueError      if CCDImage has more than one HDU (i.e. is MEF)
        Side Effect
            initialize  Policy,
                        lsst.fw.Image.CCDGeom,
                        pyfits.HDUList,
                        lsst.fw.Image.SkyRegion
        """
        Image.__init__(self)
        if (isinstance(arg, str)==True):
            self.fileName = arg
            self.hdus = pyfits.open(self.fileName, "update")
        elif (isinstance(arg, pyfits.ImageHDU)==True):
            self.hdus = pyfits.HDUList(arg)
        else:
            raise ValueError, "CCDImage must construct from a filename or a single HDU"

        if (len(self.hdus) > 1):
            raise ValueError, "CCDImage must contain only a single HDU"

        if ( not policyFile ):
            conf_file = "CCDImage.conf"
        else:
            conf_file = policyFile

        self.policy = Policy(conf_file, kws)

        geom = self.policy.Get('geom')
        if geom:
            (xoff, yoff, rot, xflip, yflip) = geom
            self.geom = CCDGeom(xoff, yoff, rot, xflip, yflip)
        else:
            print "Warning: no geometry from parent mosaic"
            self.geom = None

        #
        # metaData will be initialized with a dictionary of metadata items needed
        # elsewhere in the application, handling the variety of aliases for each
        # that may be present in the particular fits header
        #
        self.metaData = ImageMetaData(self.GetHeader())
        #
        # Now we can finish the setup of the CCDGeom object
        #
        if (self.geom):
            self.geom.setSkyParams(self.metaData.GetKW('posAngle'), self.metaData.GetKW('pixScale'))

        if (self.policy.conf.has_key('RegionFuzz')):
            self.BuildSkyRegion(self.policy.conf['RegionFuzz'])
        else:
            self.BuildSkyRegion()
        return


    #---------------------------------------------------
    def _FixHeader(self):
        """
        If this CCDImage was constructed from an HDU from an MEF file, it
        will not have the proper keywords to be a primary HDU.  Fix that.

        Input
            none
        Return
            none
        Side Effect
            Header keywords updated as primary HDU
        """
        hdr = self.GetHeader()
        if (hdr.has_key('XTENSION')):
            hdr.update('SIMPLE', 'T', before="XTENSION")
            hdr.update('EXTEND', 'F', after="NAXIS2")
            del hdr['XTENSION']
            del hdr['PCOUNT']
            del hdr['GCOUNT']
        return

    #---------------------------------------------------
    def ExtractSources(self):
        """
        Run sextractor to identify the sources in the image. The sextractor
        parameters will be taken from the policy file.  An
        ImageSourceCollection is returned.

        Input
            None
        Return
            ImageSourceCollection   containing the sources from CCDImage
        Output
            Temp FITS file named by keyword: 'ExtractSources.tmpImageFile'.
            A handful of other sextractor temp files, too.
        """
        #
        # Write the image to a temporary file (this will go away in the future)
        # Get the policy for sextractor
        # Run sextractor
        # The resulting sexcatalog becomes part of the CCDImage (?)

        self.tmpFileName = self.policy.conf['ExtractSources.tmpImageFile']

        # Kluge required to make legal fits file from single HDU from MEF
        # might want to do a verify before the FixHeader...

        self._FixHeader()
        self.hdus.writeto(self.tmpFileName, output_verify='ignore', clobber=True)
        self.sex = SExtractor()
        #
        # set up the sextractor configs here...
        # sex.config['whatever'] = something
        #
        # THRESH_TYPE not implemented in sextractor yet
#        self.sex.config['THRESH_TYPE'] = 'RELATIVE';

        self.sex.config['PIXEL_SCALE'] = self.metaData.GetKW('pixScale')
        self.sex.config['DETECT_THRESH'] = self.policy.conf['ExtractSources.DETECT_THRESH']
        self.sex.config['SEEING_FWHM'] = self.policy.conf['ExtractSources.SEEING_FWHM']
        self.sex.config['DETECT_MINAREA'] = self.policy.conf['ExtractSources.DETECT_MINAREA']
        self.sex.run(self.tmpFileName)
        return ImageSourceCollection(self.sex.catalog(), geom=self.geom)


    #---------------------------------------------------
    def BuildSkyRegion(self, fuzzDegrees=None, wcs=None):
        """
        Calculate the bounding region on the sky for this image.  To avoid
        problems near the pole, the region is a circle rather than a rectangle
        in (ra,dec).

        Input
            fuzzDegrees     bounding region around the CCDImage's FOV.
                            Format: float; Default: none
            wcs             if supplied, the wcs for this ccd will be used to calculate the SkyRegion
        Return
            none
        SideEffect
            SkyRegion is defined for this.CCDImage
        """
        #
        # Rough diameter of circle on sky
        #
        pixExtent = math.sqrt(self.hdus[0].header['NAXIS1']**2 + self.hdus[0].header['NAXIS2']**2)
        skyRadius = 0.5 * pixExtent * self.metaData.GetKW('pixScale') / 3600.0
        if (fuzzDegrees == None):
            fuzzDegrees = self.policy.conf['BuildSkyRegion.fuzzDegrees']
        if (fuzzDegrees > 0):
            skyRadius = skyRadius + fuzzDegrees
        #
        # Build the SkyRegion and return
        #
        if wcs:
            pixXCenter = 0.5*self.hdus[0].header['NAXIS1']
            pixYCenter = 0.5*self.hdus[0].header['NAXIS2']
            (ra, dec) = wcs.Pix2WCS(pixXCenter, pixYCenter)
        else:
            (ra, dec) = (self.metaData.GetKW('raDeg'), self.metaData.GetKW('decDeg'))

        self.skyRegion = SkyRegion(ra, dec, skyRadius)

        return

    #---------------------------------------------------
    def GetSkyRegion(self):
        """
        GetSkyRegion returns the SkyRegion for the CCDImage

        Return
            SkyRegion   SkyRegion overlapping the CCDImage FOV;
                        Format: SkyRegion; Default: none
        """
        return self.skyRegion

    #---------------------------------------------------
    def GetData(self):
        """
        GetData returns the data section for the CCDImage

        Return
            data        data extracted from the CCDImage
                        Format: numarray; Default: none
        """
        return self.hdus[0].data

    #---------------------------------------------------
    def GetHeader(self):
        """
        GetHeader returns the HDU for the CCDImage

        Return
            hdu         HDU for  the CCDImage;
                        Format: dictionary; Default: none
        """
        return self.hdus[0].header

    #---------------------------------------------------
    def GetMetaData(self):
        """
        GetMetaData returns the metadata for the CCDImage

        Return
            metadata    metadata for the CCDImage
                        Format: ImageMetadata object; Default: none
        """
        return self.metaData

    #---------------------------------------------------
    def Display(self, ds9win):
        """
        Using the RO.DS9 class, display the image on a ds9 window.

        Input
            ds9win
        Return
            none
        Raise
            ValueError if ds9 window not open
        """
        if (not ds9win.isOpen()):
            raise ValueError, "Open DS9Win must be supplied"
        ds9win.showArray(self.GetData())
        return


