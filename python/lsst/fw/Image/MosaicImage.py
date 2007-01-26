__all__ = ["MosaicImage"]

from Image import Image

class MosaicImage(Image):
    def __init__(self, arg, mosaicPolicyFile=None, ccdPolicyFile=None, **kws):
        if self._isFic(arg):
            from FICMosaicImage import FICMosaicImage
            self.O = FICMosaicImage(arg, mosaicPolicyFile, ccdPolicyFile, **kws)
        else:
            from FITSMosaicImage import FITSMosaicImage
            self.O = FITSMosaicImage(arg, mosaicPolicyFile, ccdPolicyFile, **kws)

    def CalculateSkyRegions(self, fuzzDegrees=None):
        self.O.CalculateSkyRegions(fuzzDegrees)

    def GetCCDImage(self, which):
        return self.O.GetCCDImage(which)

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
