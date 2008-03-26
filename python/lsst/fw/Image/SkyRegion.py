__all__ = ["SkyRegion"]

import math

class SkyRegion:
    #------------------------------------------------------
    def __init__(self, raDeg, decDeg, radiusDeg):
        """
        __init__ SkyRegion initialization

        Input
            raDeg       RA of center of SkyRegion (degrees)
                        Format: float; Default: none
            decDeg      Declination center of SkyRegion (degrees)
                        Format: float; Default: none
            radiusDeg   radius of bounding area around (raDeg,decDeg) (degrees); 
                        Format: float; Default: none
        Return
            none
        """
        self.raDeg = raDeg
        self.decDeg = decDeg
        self.radiusDeg = radiusDeg

    #------------------------------------------------------
    def RaLimits(self):
        """
        RaLimits returns the minimum and maximum range of RA in SkyRegion

        Input
            none
        Output
            (minRA,maxRA)   2 element array containing the min and max
                            range of RA in SkyRegion;
                            Format: float vector; Default: none.
        """
        (decMinus, decPlus) = self.DecLimits()
        decMax = max(abs(decMinus), abs(decPlus))
        if (decMax == 90.0):
            self.deltaRa = 180.0
        else:
            self.deltaRa = min(self.radiusDeg/ \
                        math.cos(self.decDeg*math.pi/180.0), 180.0)
            
        return [self.raDeg - self.deltaRa, self.raDeg + self.deltaRa]

    #------------------------------------------------------
    def DecLimits(self):
        """
        DecLimits returns the minimum and maximum range of Dec in SkyRegion

        Input
            none
        Output
            (minDec,maxDec) 2 element array containing the min and max
                            range of Dec in SkyRegion;
                            Format: float vector; Default: none.
        """
        return [max(self.decDeg - self.radiusDeg, -90.0), \
                min(self.decDeg + self.radiusDeg, 90.0)]

    #------------------------------------------------------
    def DeltaRa(self):
        """
        DeltaRa returns the minimum and maximum range of RA in SkyRegion

        Input
            none
        Output
            (minRA,maxRA)   2 element array containing the min and max
                            range of RA in SkyRegion;
                            Format: float vector; Default: none.
        """
        self.RaLimits()
        return self.deltaRa

    #------------------------------------------------------
    def DeltaDec(self):
        """
        DeltaDec returns the minimum and maximum range of Dec in SkyRegion

        Input
            none
        Output
            (minDec,maxDec) 2 element array containing the min and max
                            range of Dec in SkyRegion; 
                            Format: float vector; Default: none.
        """
        return self.radiusDeg

    #------------------------------------------------------
    def GetRa(self):
        """
        GetRa returns RA in SkyRegion

        Input
            none
        Output
            RA         RA of SkyRegion;
                       Format: float ; Default: none.
        """
        return self.raDeg

    #------------------------------------------------------
    def GetDec(self):
        """
        GetDec return Dec in SkyRegion

        Input
            none
        Output
            Dec         Dec of SkyRegion;
                        Format: float vector; Default: none.
        """
        return self.decDeg

    #------------------------------------------------------
    def GetRadius(self):
        """
        GetRadius returns radus of SkyRegion

        Input
            none
        Output
            radius          radius of SkyRegion; 
                            Format: float vector; Default: none.
        """
        return self.radiusDeg
