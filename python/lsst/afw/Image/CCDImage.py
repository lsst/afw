__all__ = ["CCDImage"]

from Image import Image

class CCDImage(Image):
    def __init__(self, arg, policyFile=None, **kws):
        if self._isFic(arg):
            from FICCCDImage import FICCCDImage
            self.O = FICCCDImage(arg, policyFile, **kws)
        else:
            from FITSCCDImage import FITSCCDImage
            self.O = FITSCCDImage(arg, policyFile, **kws)

    def ExtractSources(self):
        return self.O.ExtractSources()

    def BuildSkyRegion(self, fuzzDegrees=None, wcs=None):
        self.O.BuildSkyRegion(fuzzDegrees, wcs)

    def GetSkyRegion(self):
        return self.O.GetSkyRegion()

    def GetData(self):
        return self.O.GetData()

    def GetHeader(self):
        return self.O.GetHeader()

    def GetMetaData(self):
        return self.O.GetMetaData()

    def Display(self, ds9win):
        self.O.Display(ds9win)

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
