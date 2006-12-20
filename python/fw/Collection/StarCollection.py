#!/usr/bin/env python
"""
StarCollection

Creates a StarCollection in a specified SkyRegion from a selected 
star catalog. Uses the STSCI wcstools catalog access routines.
"""
__all__ = ["StarCollection"]

import sys
import os
import math
from numarray import *
 
from fw.Catalog.ctgread import *
from fw.Image import SkyRegion
import RO.DS9
import RO.StringUtil


class StarCollection:
    
    # -----------------------------------------------------------------
    def __init__(self, sr, catType, nStarMax=100, verbose=0, idArray=None,raArray=None,decArray=None,raPMArray=None,decPMArray=None,mag1Array=None,mag2Array=None,fluxArray=None):
        """
        Input
            sr          SkyRegion boundary for StarCollection contents;
                        Default: none
            catType     catalog type ; one of {'ty2','gsc','ub1', 'txtcat','handbuilt'}
            nStarMax    max number of stars to return; 
                        Format: integer; Default: 100
            verbose     verbosity level based on STSCI wcstools definition;
                        Format: integer [:]; Default: 0
	    idArray	Required only for 'handbuilt'; star identifier
                        Format: numarray integer [:]; Default: 0
	    raArray	Required only for 'handbuilt'; star RA
                        Format: numarray float [:]; Default: 0
	    decArray	Required only for 'handbuilt'; star Dec
                        Format: numarray float [:]; Default: 0
	    raPMArray	Required only for 'handbuilt'; star RA proper motion
                        Format: numarray float [:]; Default: 0
	    decPMArray	Required only for 'handbuilt'; star Dec proper motion
                        Format: numarray float [:]; Default: 0
	    mag1Array	Required only for 'handbuilt'; first mag 
                        Format: numarray float [:]; Default: 0
	    mag2Array	Required only for 'handbuilt'; second mag 
                        Format: numarray float [:]; Default: 0
	    fluxArray	Required only for 'handbuilt';	star flux
                        Format: numarray integer [:]; Default: 0
        Return
            none
        """
        # These codes come from wcstools/libwcs/wcscat.h
        __catCode = {'ty2':16, 'gsc':1, 'ub1':21,'txtcat':-3}
        
        #print "StarCollection: catType:%s  code:%d" % (catType,__catCode[catType])
        if (not isinstance(sr, SkyRegion)):
            raise BadSkyRegion, 'invalid SkyRegion input parameter'
        if (not catType in ['ty2', 'gsc', 'ub1','txtcat','handbuilt']):
            raise BadCatType,'invalid Catalog type provided'

        self.sr = sr
        
        if catType == 'handbuilt': # Acquire data from arrays passed to method
	    if idArray == None or raArray == None or decArray == None or raPMArray == None or decPMArray == None or mag1Array == None or mag2Array == None or fluxArray == None:
	        raise MissingParams,'missing content data for handbuilts'

	    else:
	        self.idArray = idArray
                self.raArray = raArray
                self.decArray = decArray
                self.raPMArray = raPMArray
                self.decPMArray = decPMArray
                self.mag1Array = mag1Array
                self.mag2Array = mag2Array
                self.fluxArray = fluxArray
		self.nStars = len(idArray)
	    return

        else:  # Acquire data from external catalog
            #
            # invoke ctgread to populate list of stars
            #
            # NOTE: epout needs to be taken from the image, not set to 2000!
            #
            try:
                (self.nStars, self.idArray, self.raArray, self.decArray, 
                        self.raPMArray, self.decPMArray, self.magArray, 
                        self.fluxArray) = \
                    ctgread(catType, __catCode[catType], 0, sr.GetRa(),
                        sr.GetDec(), sr.DeltaRa(), sr.DeltaDec(), sr.GetRadius(),
                        0, 1, 2000., 2000., 0, 0, 0, nStarMax, verbose)
    
                #print "StarCollection: Found %d stars\n" % (self.nStars)
        
                try:
                    # ctgread returns arrays dimensioned at nStarMax.  
                    # Resize to actual data length
                    self.idArray.resize(self.nStars)
                    self.raArray.resize(self.nStars)
                    self.decArray.resize(self.nStars)
                    self.raPMArray.resize(self.nStars)
                    self.decPMArray.resize(self.nStars)
                    self.mag1Array = self.magArray[0,:]
                    self.mag1Array.resize(self.nStars)
                    self.mag2Array = self.magArray[1,:]
                    self.mag2Array.resize(self.nStars)
                    self.fluxArray.resize(self.nStars)
                except:
                    raise sys.exc_type,'StarCollection: Failed resizing arrays for fiducial overlay stars.'
                
            except:
                raise sys.exc_type, 'StarCollection: \n\t\tEither:  Catalog not found\n\t\tOr: No catalog entries overlapping specified SkyRegion' 
    
        return
    
    # -----------------------------------------------------------------
    def SortByMag(self, whichMag=1, truncLen=None):
        """
        SortByMag    delivers the 'truncLen' brightest stars in order of 
                     decreasing brightness for the magnitude type selected.
                  
        Input
            whichMag    Which magnitude array to sort on
                        =1 : mag1Array
                        =2 : mag2Array
                        Default: none; Format: {1, 2}
            truncLen    Truncate all arrays to length specified. 
                        Default: no trucation; Format: int
                        Arrays affected:
                             idArray,raArray,decArray,raPMArray,decPMArray,
                             mag1Array,mag2Array,fluxArray
        Return
            none
        """
        if (whichMag==1):
            tempIndex = argsort(self.mag1Array,0)
        elif (whichMag==2):
            tempIndex = argsort(self.mag2Array,0)
        else:
            raise ValueError, 'invalid Magnitude type provided'
        if (truncLen):
            extrIndex = tempIndex[0:truncLen]
        else:
            extrIndex = tempIndex
        
        self.idArray = take(self.idArray, extrIndex)
        self.raArray = take(self.raArray, extrIndex)
        self.decArray = take(self.decArray, extrIndex)
        self.raPMArray = take(self.raPMArray, extrIndex)
        self.decPMArray = take(self.decPMArray, extrIndex)
        self.mag1Array = take(self.mag1Array, extrIndex)
        self.mag2Array = take(self.mag2Array, extrIndex)
        self.fluxArray = take(self.fluxArray, extrIndex)

        if (truncLen and (self.nStars > truncLen)):
            self.nStars = truncLen
            self.idArray.resize(self.nStars)
            self.raArray.resize(self.nStars)
            self.decArray.resize(self.nStars)
            self.raPMArray.resize(self.nStars)
            self.decPMArray.resize(self.nStars)
            self.mag1Array.resize(self.nStars)
            self.mag2Array.resize(self.nStars)
            self.fluxArray.resize(self.nStars)
       
        return
    
    # -----------------------------------------------------------------
    def GetXiEta(self):
        """
        Convert the (ra,dec) coordinates to tangent plane (xi, eta) 
        with center of projection given by the (ra, dec) of the SkyRegion 
        used in construction.

        We may wish to allow the projection center be arbitrary, with the 
        SkyRegion only the default

        Input       
            none
        Return
            xi          tangent plane coord for SkyRegion;
                        Units: arcsec; format: float numarray.
            eta         tangent plane coord for SkyRegion;
                        Units: arcsec; format float numarray.
            mag         Star magnitude from mag1Array
            id          id (unique for each 'row') across all arrays;
                        Format: integer.
        """

        degToRad = math.pi / 180.0
        radToArcsec = 180.0 * 3600.0 / math.pi
        
        sinStarDec = sin(degToRad*self.decArray)
        cosStarDec = cos(degToRad*self.decArray)
        cosDeltaRa = cos(degToRad*(self.raArray - self.sr.GetRa()))
        sinDeltaRa = sin(degToRad*(self.raArray - self.sr.GetRa()))
        sinDec0 = sin(degToRad*self.sr.GetDec())
        cosDec0 = cos(degToRad*self.sr.GetDec())

        fac = 1.0/( sinStarDec * sinDec0 + cosStarDec * cosDec0 * cosDeltaRa)
        self.xi = cosStarDec * sinDeltaRa * fac * radToArcsec
        self.eta = ( sinStarDec * cosDec0 - cosStarDec * sinDec0 * cosDeltaRa) * fac * radToArcsec

        return (self.xi, self.eta, self.mag1Array, arange(self.nStars))

    # -----------------------------------------------------------------
    def DisplayStars(self, ds9win, wcs=None, displayRadius=10):
        """
        DisplayStars draws the star locations on the open DS9 display, 
        using the provided wcs.

        Input
            ds9win          Open ds9 window; Default: none
            wcs             WCS object; Default: none
            displayRadius    radius of circle to draw around sources, in pixels; Format: float; Default: 10
        Return
            none
        """
        if wcs:
            for i in range(self.nStars):
                (x, y) = wcs.WCS2pix(self.raArray[i], self.decArray[i])
                ds9win.xpaset('regions', data='circle(%f,%f,%d)' % (x,y,displayRadius))
                print '%f %f (ra,dec) -> %f %f\n' % (self.raArray[i], self.decArray[i], x, y)
        else:
            for i in range(self.nStars):
                raStr = RO.StringUtil.dmsStrFromDeg(self.raArray[i]/15.0)
                decStr = RO.StringUtil.dmsStrFromDeg(self.decArray[i])
                ds9win.xpaset('regions', data='fk5; circle(%s,%s,%f\")' % (raStr,decStr,displayRadius))
        return


    # -----------------------------------------------------------------
    def WriteStarsXiEta(self, fileName):
        """
        Somewhat weird output format tuned for StarLink findoff

        Input
            filename    filename for 'xi/eta/mag' output tuned for 
                        StarLink FINDOFF
        Return
            none

        Output File
            Format: '%d %f %f %f %d\n' % (i, xi[i],eta[i],mag[i], i)
        """
        (xi, eta, mag, rangeStars) = self.GetXiEta()

        f=os.open(fileName,os.O_CREAT|os.O_RDWR)
        for i in rangeStars:
            os.write(f,'%d %f %f %f %d\n' % (i, xi[i],eta[i],mag[i], i))

        os.close(f)
                     
