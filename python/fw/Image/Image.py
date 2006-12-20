__all__ = ["Image"]

class Image:
    def __init__(self, fileName=None):
        if self._isFic(fileName):
            from FICImage import FICImage
            self.O = FICImage(fileName)
        else:
            from FITSImage import FITSImage
            self.O = FITSImage(fileName)

    def __del__(self):
        self.O.__del__()

    def NumCCDS(self):
        return self.O.NumCCDS()

    def GetCCDHDU(self, whichCCD):
        return self.O.GetCCDHDU(whichCCD)

    def GetParameter(self, paramName):
        return self.O.GetParameter(paramName)

    def SetParameter(self, paramName, paramValue):
        self.O.SetParameter(paramName, paramValue)

    def DelParameter(self, paramName):
        self.O.DelParameter(paramName)

    def _isFic(self, fileName):
        if not isinstance(fileName, str):
            return False
        if fileName is None:
            # Let FITS be the assumed default.
            return False
        if fileName[-4:] == '.fic':
            return True
        if fileName[-5:] == '.fic/':
            return True
        return False
