import pyfits
from RO.StringUtil import *

__all__ = ["ImageMetaData"]

class ImageMetaData:
    """
    ImageMetaData is initialized from a PyFITS Header object.  It
    scans the header for keywords needed to set the values of some
    key metadata items generally needed by applications.
    """
    #---------------------------------------------------------
    def __init__(self, header):
        """
        __init__ ImageMetaData initialization

        Input
            header      keyword dictionary from FITS header 'keyword=value'
                        Format: dictionary; Default: none
        Return
            none
        """
        self.metaDict = {'ra': None, 'dec': None, 'pixScale': None, 'mjd': None, \
                         'raDeg': None, 'decDeg':None, 'epoch':2000.0, 'posAngle': None}
        self.header = header
        
        self.raCandidates = ['RA', 'OBSRA']
        for raKW in self.raCandidates:
            if (self.header.has_key(raKW)):
                self.metaDict['ra'] = self.header[raKW]
                break
        if (self.metaDict['ra'] == None):
            raise ValueError, "No RA keyword in image"

        self.decCandidates = ['DEC', 'OBSDEC']
        for decKW in self.decCandidates:
            if (self.header.has_key(decKW)):
                self.metaDict['dec'] = self.header[decKW]
                break
        if (self.metaDict['dec'] == None):
            raise ValueError, "No DEC keyword in image"

        #
        # Convenience keywords with strings converted to degrees
        #

        self.metaDict['raDeg'] = 15.0*degFromDMSStr(self.metaDict['ra'])
        self.metaDict['decDeg'] = degFromDMSStr(self.metaDict['dec'])

        self.scaleCandidates = ['PIXSCAL1', 'PIXSCAL', 'SECPIX']
        for scaleKW in self.scaleCandidates:
            if (self.header.has_key(scaleKW)):
                self.metaDict['pixScale'] = self.header[scaleKW]
                break
        if (self.metaDict['pixScale'] == None):
            raise ValueError, "No SCALE keyword in image"

        self.mjdCandidates = ['MJD_OBS']
        for mjdKW in self.mjdCandidates:
            if (self.header.has_key(mjdKW)):
                self.metaDict['mjd'] = self.header[mjdKW]
                break

        # Calculate epoch from mjd here
        #if (self.metaDict['mjd'] == None):
        #    print 'Warning: No MJD keyword in image'


        # Get position angle.  If not present, leave at zero, with warning
        self.paCandidates = ['ROTANGLE', 'PA_PNT']
        for paKW in self.paCandidates:
            if (self.header.has_key(paKW)):
                self.metaDict['posAngle'] = self.header[paKW]
                break
        if (self.metaDict['posAngle'] == None):
            print 'Warning: No position angle keyword in image - set to zero'
            self.metaDict['posAngle'] = 0.0

    #---------------------------------------------------------
    def GetKW(self, kwName):
        """
        GetKW(kwName) first looks to see if the requested keyword is contained
        metaDict. If so, return that one.  If not, pass through to the 
        FITS header

        Input
            kwName      keyword to lookup
        Return
            value       value associated with keyword.
                        1st lookup in local dictionary, else
                        2cd lookup in FITS header, 
                        else return None
        """

        if (self.metaDict.has_key(kwName)):
            return self.metaDict[kwName]
        elif (self.header.has_key(kwName)):
            return self.header[kwName]
        else:
            return None
           
    
