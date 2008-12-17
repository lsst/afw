namespace det = lsst::afw::detection;


void det::BaseSourceAttributes::setAllNull(){
    safe_delete(_id);
    safe_delete(_ampExposureId);
    safe_delete(_objectId);
    safe_delete(_movingObjectId);
    safe_delete(_ra);
    safe_delete(_dec);
    safe_delete(_xFlux);
    safe_delete(_xFluxErr);
    safe_delete(_yFluxv);
    safe_delete(_yFluxErr);
    safe_delete(_raFlux);
    safe_delete(_raFluxErr);
    safe_delete(_declFlux);
    safe_delete(_declFluxErr);
    safe_delete(_xPeak);
    safe_delete(_yPeak);
    safe_delete(_raPeak);
    safe_delete(_decPeak);
    safe_delete(_xAstrom);
    safe_delete(_xAstromErr);
    safe_delete(_yAstrom);
    safe_delete(_yAstromErr);
    safe_delete(_raAstrom);
    safe_delete(_raAstromErr);
    safe_delete(_declAstrom);
    safe_delete(_declAstromErr);
    safe_delete(_taiMidPoint);
    safe_delete(_psfMag);
    safe_delete(_apMag);
    safe_delete(_modelMag);
    safe_delete(_petroMag);
    safe_delete(_raErr4detection);
    safe_delete(_decErr4detection);
    safe_delete(_raErr4wcs);
    safe_delete(_decErr4wcs);
    safe_delete(_taiRange);
    safe_delete(_fwhmA);
    safe_delete(_fwhmB);
    safe_delete(_fwhmTheta);
    safe_delete(_psfMagErr);
    safe_delete(_apMagErr);  
    safe_delete(_instMag);
    safe_delete(_instMagErr);
    safe_delete(_nonGrayCorrMag);
    safe_delete(_nonGrayCorrMagErr);
    safe_delete(_atmCorrMag);
    safe_delete(_atmCorrMagErr);
    safe_delete(_apDia);
    safe_delete(_snr);
    safe_delete(_chi2);
    safe_delete(_procHistoryId);
    safe_delete(_flag4association);
    safe_delete(_flag4detection);
    safe_delete(_flag4wcs);
    safe_delete(_filterId);   
}

void det::BaseSourceAttributes::setAllNotNull(){
    setNotNull(_id);
    setNotNull(_ampExposureId);
    setNotNull(_objectId);
    setNotNull(_movingObjectId);
    setNotNull(_ra);
    setNotNull(_dec);
    setNotNull(_xFlux);
    setNotNull(_xFluxErr);
    setNotNull(_yFluxv);
    setNotNull(_yFluxErr);
    setNotNull(_raFlux);
    setNotNull(_raFluxErr);
    setNotNull(_declFlux);
    setNotNull(_declFluxErr);
    setNotNull(_xPeak);
    setNotNull(_yPeak);
    setNotNull(_raPeak);
    setNotNull(_decPeak);
    setNotNull(_xAstrom);
    setNotNull(_xAstromErr);
    setNotNull(_yAstrom);
    setNotNull(_yAstromErr);
    setNotNull(_raAstrom);
    setNotNull(_raAstromErr);
    setNotNull(_declAstrom);
    setNotNull(_declAstromErr);
    setNotNull(_taiMidPoint);
    setNotNull(_psfMag);
    setNotNull(_apMag);
    setNotNull(_modelMag);
    setNotNull(_petroMag);
    setNotNull(_raErr4detection);
    setNotNull(_decErr4detection);
    setNotNull(_raErr4wcs);
    setNotNull(_decErr4wcs);
    setNotNull(_taiRange);
    setNotNull(_fwhmA);
    setNotNull(_fwhmB);
    setNotNull(_fwhmTheta);
    setNotNull(_psfMagErr);
    setNotNull(_apMagErr);  
    setNotNull(_instMag);
    setNotNull(_instMagErr);
    setNotNull(_nonGrayCorrMag);
    setNotNull(_nonGrayCorrMagErr);
    setNotNull(_atmCorrMag);
    setNotNull(_atmCorrMagErr);
    setNotNull(_apDia);
    setNotNull(_snr);
    setNotNull(_chi2);
    setNotNull(_procHistoryId);
    setNotNull(_flag4association);
    setNotNull(_flag4detection);
    setNotNull(_flag4wcs);
    setNotNull(_filterId);   
}

det::BaseSourceAttributes::~BaseSourceAttributes() {
    setAllNull();
}


bool det::BaseSourceAttributes::operator==(BaseSourceAttributes const & d) const {
    if (this == &d)  {
        return true;
    }
    
    if ( areEqual(_sourceId, d._sourceId)                   &&
         areEqual(_filterId, d._filterId)                   &&
         areEqual(_procHistoryId, d._procHistoryId)         &&
         areEqual(_ra, d._ra)                               &&
         areEqual(_dec, d._dec)                             &&
         areEqual(_raErr4Wcs, d._raErr4wcs)                 &&
         areEqual(_decErr4Wcs, d._decErr4wcs)               &&
         areEqual(_taiMidPoint, d._taiMidPoint)             &&
         areEqual(_fwhmA, d._fwhmA)                         &&
         areEqual(_fwhmB, d._fwhmB)                         &&
         areEqual(_fwhmTheta, d._fwhmTheta)                 &&
         areEqual(_psfMag, d._psfMag)                       &&
         areEqual(_psfMagErr, d._psfMagErr)                 &&
         areEqual(_apMag, d._apMag)                         &&
         areEqual(_apMagErr, d._apMagErr)                   &&
         areEqual(_modelMag, d._modelMag)                   &&
         areEqual(_modelMagErr, d._modelMagErr)             &&
         areEqual(_instMag, d._instMag)                     &&
         areEqual(_instMagErr, d._instMagErr)               &&
         areEqual(_nonGrayCorrMag, d._nonGrayCorrMag)       &&
         areEqual(_nonGrayCorrMagErr, d._nonGrayCorrMagErr) &&
         areEqual(_atmCorrMag, d._atmCorrMag)               &&
         areEqual(_atmCorrMagErr, d._atmCorrMagErr)         &&
         areEqual(_snr, d._snr)                             &&
         areEqual(_chi2, d._chi2)                           &&
         areEqual(_ampExposureId, d._ampExposureId)         &&
         areEqual(_objectId, d._objectId)                   &&
         areEqual(_movingObjectId, d._movingObjectId)       &&
         areEqual(_raErr4Detection, d._raErr4Detection)     &&
         areEqual(_decErr4Detection, d._decErr4Detection)   &&
         areEqual(_xFlux, d._xFlux)                         &&
         areEqual(_xFluxErr, d._xFluxErr)                   &&
         areEqual(_yFlux, d._yFlux)                         &&
         areEqual(_yFluxErr, d._yFluxErr)                   &&
         areEqual(_xPeak, d._xPeak)                         && 
         areEqual(_yPeak, d._yPeak)                         && 
         areEqual(_raPeak, d._raPeak)                       && 
         areEqual(_decPeak, d._decPeak)                     &&   
         areEqual(_xAstrom, d._xAstrom)                     &&
         areEqual(_xAstromErr, d._xAstromErr)               &&                   
         areEqual(_yAstrom, d._yAstrom)                     &&
         areEqual(_yAstromErr, d._yAstromErr)               &&                                   
         areEqual(_raAstrom, d._raAstrom)                   &&
         areEqual(_raAstromErr, d._raAstromErr)             &&                   
         areEqual(_decAstrom, d._decAstrom)                 &&
         areEqual(_decAstromErr, d._decAstromErr)           &&                                   
         areEqual(_taiRange, d._taiRange)                   &&
         areEqual(_apDia, d._apDia)                         &&
         areEqual(_flag4association, d._flag4association)   &&
         areEqual(_flag4detection, d._flag4detection)       &&
         areEqual(_flag4wcs, d._flag4wcs) ){
         return true;     
    }
    
    return false;
}

template <typename Archive> 
void det::BaseSourceAttributes::serialize(Archive & ar, unsigned int const version) {
    serialzeData(ar, version, _id);
    serialzeData(ar, version, _ampExposureId);
    serialzeData(ar, version, _filterId);
    serialzeData(ar, version, _objectId);
    serialzeData(ar, version, _movingObjectId);
    serialzeData(ar, version, _procHistoryId);
    serialzeData(ar, version, _ra);
    serialzeData(ar, version, _dec);
    serialzeData(ar, version, _raErr4detection);
    serialzeData(ar, version, _decErr4detection);
    serialzeData(ar, version, _raErr4wcs);
    serialzeData(ar, version, _decErr4wcs);
    serialzeData(ar, version, _xFlux);
    serialzeData(ar, version, _xFluxErr);
    serialzeData(ar, version, _yFlux);
    serialzeData(ar, version, _yFluxErr);
    serialzeData(ar, version, _raFlux);
    serialzeData(ar, version, _raFluxErr);
    serialzeData(ar, version, _decFlux);
    serialzeData(ar, version, _decFluxErr);        
    serialzeData(ar, version, _xPeak);
    serialzeData(ar, version, _yPeak);
    serialzeData(ar, version, _raPeak);
    serialzeData(ar, version, _decPeak);
    serialzeData(ar, version, _xAstrom);
    serialzeData(ar, version, _xAstromErr);
    serialzeData(ar, version, _yAstrom);
    serialzeData(ar, version, _yAstromErr);
    serialzeData(ar, version, _raAstrom);
    serialzeData(ar, version, _raAstromErr);
    serialzeData(ar, version, _decAstrom);
    serialzeData(ar, version, _decAstromErr);        
    serialzeData(ar, version, _taiMidPoint);
    serialzeData(ar, version, _taiRange);
    serialzeData(ar, version, _fwhmA);
    serialzeData(ar, version, _fwhmB);
    serialzeData(ar, version, _fwhmTheta);
    serialzeData(ar, version, _psfMag);
    serialzeData(ar, version, _psfMagErr);
    serialzeData(ar, version, _apMag);
    serialzeData(ar, version, _apMagErr);
    serialzeData(ar, version, _modelMag);
    serialzeData(ar, version, _modelMagErr);
    serialzeData(ar, version, _instMag);
    serialzeData(ar, version, _instMagErr);
    serialzeData(ar, version, _nonGrayCorrMag);
    serialzeData(ar, version, _nonGrayCorrMagErr);
    serialzeData(ar, version, _atmCorrMag);
    serialzeData(ar, version, _atmCorrMagErr);
    serialzeData(ar, version, _apDia);        
    serialzeData(ar, version, _snr);
    serialzeData(ar, version, _chi2);
    serialzeData(ar, version, _flag4association);
    serialzeData(ar, version, _flag4detection);
    serialzeData(ar, version, _flag4wcs);
}

