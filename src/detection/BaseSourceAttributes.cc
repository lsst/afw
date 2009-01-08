namespace det = lsst::afw::detection;

template <typename Archive> 
void det::BaseSourceAttributes::serialize(Archive & ar, unsigned int const version) {
    ar & _id;
    ar & _ampExposureId;
    ar & _filterId;
    ar & _objectId;
    ar & _movingObjectId;
    ar & _procHistoryId;
    ar & _ra;
    ar & _dec;
    ar & _raErr4detection;
    ar & _decErr4detection;
    ar & _raErr4wcs;
    ar & _decErr4wcs;
    ar & _xFlux;
    ar & _xFluxErr;
    ar & _yFlux;
    ar & _yFluxErr;
    ar & _raFlux;
    ar & _raFluxErr;
    ar & _decFlux;
    ar & _decFluxErr;        
    ar & _xPeak;
    ar & _yPeak;
    ar & _raPeak;
    ar & _decPeak;
    ar & _xAstrom;
    ar & _xAstromErr;
    ar & _yAstrom;
    ar & _yAstromErr;
    ar & _raAstrom;
    ar & _raAstromErr;
    ar & _decAstrom;
    ar & _decAstromErr;
    ar & _taiMidPoint;
    ar & _taiRange;
    ar & _fwhmA;
    ar & _fwhmB;
    ar & _fwhmTheta;
    ar & _psfMag;
    ar & _psfMagErr;
    ar & _apMag;
    ar & _apMagErr;
    ar & _modelMag;
    ar & _modelMagErr;
    ar & _instMag;
    ar & _instMagErr;
    ar & _nonGrayCorrMag;
    ar & _nonGrayCorrMagErr;
    ar & _atmCorrMag;
    ar & _atmCorrMagErr;
    ar & _apDia;        
    ar & _snr;
    ar & _chi2;
    ar & _flag4association;
    ar & _flag4detection;
    ar & _flag4wcs;
    
    bool b;
    if (Archive::is_loading::value) {
        for (int i = 0; i < numNullableFields; ++i) {
            ar & b;
            _nulls.set(i, b);
        }
    } else {
        for (int i = 0; i < numNullableFields; ++i) {
            b = isNull(i);
            ar & b;
        }
    }    
}

