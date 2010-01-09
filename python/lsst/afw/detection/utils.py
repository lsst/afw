import detectionLib as afwDetect

def writeFootprintAsDefects(fd, foot):
    """Write foot as a set of Defects to fd"""

    bboxes = afwDetect.footprintToBBoxList(foot)
    for bbox in bboxes:
        print >> fd, """\
Defects: {
    x0:     %4d                         # Starting column
    width:  %4d                         # number of columns
    y0:     %4d                         # Starting row
    height: %4d                         # number of rows
}""" % (bbox.getX0(), bbox.getWidth(), bbox.getY0(), bbox.getHeight())

def makeDiaSourceFromSource(source):
    diaSource = afwDetect.DiaSource()
    diaSource.setId(source.getId())
    diaSource.setAmpExposureId(source.getAmpExposureId())
    diaSource.setFilterId(source.getFilterId())
    diaSource.setObjectId(source.getObjectId())
    diaSource.setMovingObjectId(source.getMovingObjectId())
    diaSource.setProcHistoryId(source.getProcHistoryId())
    diaSource.setXAstrom(source.getXAstrom())
    diaSource.setXAstromErr(source.getXAstromErr())
    diaSource.setYAstrom(source.getYAstrom())
    diaSource.setYAstromErr(source.getYAstromErr())
    diaSource.setRa(source.getRa())
    diaSource.setRaErrForDetection(source.getRaErrForDetection())
    diaSource.setRaErrForWcs(source.getRaErrForWcs())
    diaSource.setDec(source.getDec())
    diaSource.setDecErrForDetection(source.getDecErrForDetection())
    diaSource.setDecErrForWcs(source.getDecErrForWcs())
    diaSource.setXFlux(source.getXFlux())
    diaSource.setXFluxErr(source.getXFluxErr()) 
    diaSource.setYFlux(source.getYFlux())
    diaSource.setYFluxErr(source.getYFluxErr())
    diaSource.setRaFlux(source.getRaFlux())
    diaSource.setRaFluxErr(source.getRaFluxErr())
    diaSource.setDecFlux(source.getDecFlux())
    diaSource.setDecFluxErr(source.getDecFluxErr())
    diaSource.setXPeak(source.getXPeak())
    diaSource.setYPeak(source.getYPeak())
    diaSource.setRaPeak(source.getRaPeak())
    diaSource.setDecPeak(source.getDecPeak())
    diaSource.setRaAstrom(source.getRaAstrom())
    diaSource.setRaAstromErr(source.getRaAstromErr())
    diaSource.setDecAstrom(source.getDecAstrom())
    diaSource.setDecAstromErr(source.getDecAstromErr())
    diaSource.setTaiMidPoint(source.getTaiMidPoint())
    diaSource.setTaiRange(source.getTaiRange())
    diaSource.setPsfFlux(source.getPsfFlux())
    diaSource.setPsfFluxErr(source.getPsfFluxErr())
    diaSource.setApFlux(source.getApFlux())
    diaSource.setApFluxErr(source.getApFluxErr())
    diaSource.setModelFlux(source.getModelFlux())
    diaSource.setModelFluxErr(source.getModelFluxErr())
    diaSource.setInstFlux(source.getInstFlux())
    diaSource.setInstFluxErr(source.getInstFluxErr())
    diaSource.setNonGrayCorrFlux(source.getNonGrayCorrFlux())
    diaSource.setNonGrayCorrFluxErr(source.getNonGrayCorrFluxErr())
    diaSource.setAtmCorrFlux(source.getAtmCorrFlux())
    diaSource.setAtmCorrFluxErr(source.getAtmCorrFluxErr())
    diaSource.setApDia(source.getApDia()) 
    diaSource.setIxx(source.getIxx())
    diaSource.setIxxErr(source.getIxxErr())
    diaSource.setIyy(source.getIyy())
    diaSource.setIyyErr(source.getIyyErr())
    diaSource.setIxy(source.getIxy())
    diaSource.setIxyErr(source.getIxyErr())
    diaSource.setSnr(source.getSnr())
    diaSource.setChi2(source.getChi2())
    diaSource.setFlagForAssociation(source.getFlagForAssociation())
    diaSource.setFlagForDetection(source.getFlagForDetection())
    diaSource.setFlagForWcs(source.getFlagForWcs())

    for i in xrange(afwDetect.NUM_SHARED_NULLABLE_FIELDS):
        diaSource.setNull(i, source.isNull(i))

    return diaSource
