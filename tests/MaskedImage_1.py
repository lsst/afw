import lsst.fw.Core.fwLib as fwCore
import fwTests

def test():
    testMaskedImage1 = fwCore.MaskedImageD(272, 1037)
    testMaskedImage2 = fwCore.MaskedImageD(272, 1037)

    for tm in (testMaskedImage1, testMaskedImage2):
        tm.getMask().addMaskPlane("CR")

    testMaskedImage2 += testMaskedImage1;

    fooFunc = fwTests.testPixProcFuncD(testMaskedImage1)

    fooFunc.init()
    testMaskedImage1.processPixels(fooFunc)

    fooFunc.init()
    testMaskedImage1.processPixels(fooFunc)

if __name__ == "__main__":
    test()
