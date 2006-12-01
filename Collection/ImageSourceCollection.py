########################################
"""
ImageSourceCollection

Creates an ImageSourceCollection using "sextractor"
"""
__all__ = ["ImageSourceCollection"]

import os
import sys
import numarray

 
class ImageSourceCollection:

    # -----------------------------------------------------------------
    def __init__(self, sourceInfo, geom=None):
        """
        ImageSourceCollection initialization

        Input
            sourceInfo      sourceInfo is expected to be list of dictionaries;
                            returned by sexcatalog; Error if not.
            geom            CCDGeom - containing geometry of source image;
                            Default: none
        Return
            none
        """
        if (isinstance(sourceInfo, list)==False):
            raise(DataError)
        if ((not sourceInfo[0].has_key('XWIN_IMAGE')) or (not sourceInfo[0].has_key('YWIN_IMAGE')) or
            (not sourceInfo[0].has_key('FLUX_BEST')) or (not sourceInfo[0].has_key('FLAGS')) or
            (not sourceInfo[0].has_key('CLASS_STAR'))):
            raise(DataError)
        self.nSources = len(sourceInfo)
        self.sourceArray = numarray.zeros([self.nSources,5], type='Float32')
        self.flagArray= numarray.zeros([self.nSources], type='UInt8')
        i = 0
        for source in sourceInfo:
            self.sourceArray[i] = [source['XWIN_IMAGE'], source['YWIN_IMAGE'], source['FLUX_BEST'], source['CLASS_STAR'], i]
            self.flagArray[i] = source['FLAGS']
            i = i + 1

        self.geom = geom

    # -----------------------------------------------------------------
    def SortByFlux(self):
        """
        SortByFlux  sorts the source stars by flux

        Input
            none

        Return
            none

        Side Effect
            self.sourceArray is sorted by flux
        """
        self.ind = numarray.argsort(-self.sourceArray[:,2], 0)
        self.sourceArray = numarray.take(self.sourceArray, self.ind)
        #
        # Note, index recomputed with new sorted order, because later users
        # will use it to index directly
        self.sourceArray[:,4] = numarray.arange(self.nSources)
        
    # -----------------------------------------------------------------
    def GetSources(self, flagMask=None, starGalCut=0):
        """
        Return numarrays containing the x, y, and mag.
        If asked, filter using flagMask.o

        Input
            flagMask        If set, then return only sources that for which the sextractor flags
                            identified by one bits in flagMask are not set
                            Format: unsigned int; Default: None

            starGalCut      If set, then return only sources for which the sextractor estimate of whether
                            the source is a star or a galaxy, CLASS_STAR, is >= starGalCut
                            Format:     Default: 0
        Return
            x               x pixel location on source image;
                            Format: float numarray; Default: none.
            y               y pixel location on source image;
                            Fromat: float numarray; Default: none.
            mag             magnitude of source image at (x,y) pixel 
                            Format: float numarray;   Default: none

        """
        ind = numarray.where(numarray.logical_and(self.sourceArray[:,2] > 0, self.sourceArray[:,3] >= starGalCut), 1, 0)
        self.SortByFlux()
        
        if (flagMask != None):
            ind = numarray.where(numarray.logical_and(self.flagArray & flagMask == 0, ind == 1), 1, 0)

        tmpSources = numarray.compress(ind,self.sourceArray, 0)
        return [tmpSources[:,0], tmpSources[:,1], -2.5*numarray.log10(tmpSources[:,2])]

    # -----------------------------------------------------------------
    def GetSourcesXiEta(self, flagMask=None, starGalCut=0, takeTop=None):
        """
        Return numarrays containing the xi, eta, and ???.
        If asked, filter using flagMask.o

        Input
            flagMask        If set, then return only sources that for which the sextractor flags
                            identified by one bits in flagMask are not set
                            Format: unsigned int; Default: None

            starGalCut      If set, then return only sources for which the sextractor estimate of whether
                            the source is a star or a galaxy, CLASS_STAR, is >= starGalCut
                            Format:     Default: 0
            takeTop         After filtering and cutting, truncate remaining
                            sources to no more than takeTop.
                            Format: integer;    Default: none
        Return
            xi              xi on tangent plane of source image;
                            Format: float numarray; Default: none.
            eta             eta tangent plane of on source image;
                            Format: float numarray; Default: none.
            mag             instrumental mag of source image at (xi,eta) location 
                            Format: float numarray;   Default: none
            id              Source id
                            Format: float numarray;  Default: none

        """
        if (not self.geom):
            raise LookupError,"No CCDGeom for Source Image"

        self.SortByFlux()
        
        ind = numarray.where(numarray.logical_and(self.sourceArray[:,2] > 0, self.sourceArray[:,3] >= starGalCut), 1, 0)
        
        if (flagMask != None):
            ind = numarray.where(numarray.logical_and(self.flagArray & flagMask == 0, ind == 1), 1, 0)

        tmpSources = numarray.compress(ind,self.sourceArray, 0)

        if takeTop:
            tmpSources = numarray.resize(tmpSources,[int(takeTop), 5])

        for i in range(len(tmpSources)):
            (tmpSources[i,0], tmpSources[i,1]) = self.geom.xyToXiEta(tmpSources[i,0], tmpSources[i,1])

        return [tmpSources[:,0], tmpSources[:,1], -2.5*numarray.log10(tmpSources[:,2]), tmpSources[:,4]]

    # -----------------------------------------------------------------
    def DisplaySources(self, ds9win, flagMask=None, starGalCut=0, displayRadius=10):
        """
        DisplayStars draws the star locations on the open DS9 display,
        using the provided wcs.
                                                                                
        Input
            ds9win          Open ds9 window; Default: none.
            flagMask        If set, then return only sources that for which the sextractor flags
                            identified by one bits in flagMask are not set
                            Format: unsigned int; Default: None

            starGalCut      If set, then return only sources for which the sextractor estimate of whether
                            the source is a star or a galaxy, CLASS_STAR, is >= starGalCut
                            Format:     Default: 0
            displayRadius   radius of circle to draw around sources, in pixels; Format: float; Default: 10
        Return
            none

        """
        if (not ds9win.isOpen()):
            raise ValueError, "Open DS9Win must be supplied"

        ind = numarray.where(numarray.logical_and(self.sourceArray[:,2] > 0, self.sourceArray[:,3] >= starGalCut), 1, 0)
        
        if (flagMask != None):
            ind = numarray.where(numarray.logical_and(self.flagArray & flagMask == 0, ind == 1), 1, 0)

 
        tmpSources = numarray.compress(ind,self.sourceArray, 0)

        for source in tmpSources:
            ds9win.xpaset('regions', data='box(%f,%f,%d, %d, 0)' % (source[0],source[1],displayRadius,displayRadius))


    # -----------------------------------------------------------------
    def WriteSourcesXiEta(self, fileName, takeTop=None):
        """
        Somewhat weird output format tuned for StarLink findoff
                                                                                
        Input
            fileName    filename for 'xi/eta/mag' output tuned for
                        StarLink FINDOFF
            takeTop     max count of sources to requested
        Return
            none
                                                                                
        Output File
            Format: '%d %f %f %f %d\\n' % (i, xi[i],eta[i],mag[i], id[i])

        """
        try:
            (xi, eta, mag, id) = self.GetSourcesXiEta(takeTop=takeTop)

        except:
            raise sys.exc_type,sys.exc_value
            

        try:
            f=os.open(fileName,os.O_CREAT|os.O_RDWR)
        except:
            print sys.exc_type,"\nCause: ",sys.exc_value,"\n"
            raise IOError, "Failed to open temp file:" + fileName

        for i in range(len(xi)):
            os.write(f,'%d %f %f %f %d\n' % (i, xi[i],eta[i],mag[i], id[i]))
        os.close(f)

                     
        
