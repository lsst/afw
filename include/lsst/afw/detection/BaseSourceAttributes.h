#ifndef LSST_AFW_DETECTION_BASE_SOURCE_ATTRIBUTES_H
#define LSST_AFW_DETECTION_BASE_SOURCE_ATTRIBUTES_H

#include <bitset>
#include "boost/cstdint.hpp"

namespace boost {
namespace serialization {
    class access;
}}

namespace lsst{
namespace afw {
namespace detection {

/**
 * List of NULLABLE fields common to all source types
 */
enum SharedNullableField {
    OBJECT_ID,
    MOVING_OBJECT_ID,
    X_FLUX,
    X_FLUX_ERR,
    Y_FLUX,
    Y_FLUX_ERR,
    RA_FLUX,
    RA_FLUX_ERR,
    DEC_FLUX,
    DEC_FLUX_ERR,
    X_PEAK,
    Y_PEAK,
    RA_PEAK,
    DEC_PEAK,
    X_ASTROM,
    X_ASTROM_ERR,
    Y_ASTROM,
    Y_ASTROM_ERR,
    RA_ASTROM,
    RA_ASTROM_ERR,
    DEC_ASTROM,
    DEC_ASTROM_ERR,
    NON_GRAY_CORR_MAG,
    NON_GRAY_CORR_MAG_ERR,
    ATM_CORR_MAG,        
    ATM_CORR_MAG_ERR,
    AP_DIA,       
    FLAG_FOR_ASSOCIATION,
    FLAG_FOR_DETECTION,
    FLAG_FOR_WCS,
    NUM_SHARED_NULLABLE_FIELDS
};

/**
 * Base class for Sources
 * Defines all shared fields.
 * Provides functionality for supporting
 *   NULLABLE fields. Derived classes must specify the number of NULLABLE fields
 *   in the template argument for BaseSourceAttributes
 */
template<int numNullableFields>
class BaseSourceAttributes {
public:    
    virtual ~BaseSourceAttributes(){};
    
    // getters
    boost::int64_t getId() const { return _id; }
    boost::int64_t getAmpExposureId() const { return _ampExposureId; }
    boost::int8_t  getFilterId() const { return _filterId; }
    boost::int64_t getObjectId() const { return _objectId; }
    boost::int64_t getMovingObjectId() const { return _movingObjectId; }
    boost::int32_t getProcHistoryId() const { return _procHistoryId; }
    double  getRa() const { return _ra; }
    double  getDec() const { return _dec; }
    float   getRaErrForWcs() const { return _raErrForWcs; }
    float   getDecErrForWcs() const { return _decErrForWcs; }
    float   getRaErrForDetection() const { return _raErrForDetection; }
    float   getDecErrForDetection() const { return _decErrForDetection; }
    double  getXFlux() const { return _xFlux; }
    double  getXFluxErr() const { return _xFluxErr; }
    double  getYFlux() const { return _yFlux; }
    double  getYFluxErr() const { return _yFluxErr; }
    double  getRaFlux() const { return _raFlux; }
    double  getRaFluxErr() const { return _raFluxErr; }
    double  getDecFlux() const { return _decFlux; }
    double  getDecFluxErr() const { return _decFluxErr; }
    double  getXPeak() const { return _xPeak; }
    double  getYPeak() const { return _yPeak; }
    double  getRaPeak() const { return _raPeak; }
    double  getDecPeak() const { return _decPeak; }
    double  getXAstrom() const { return _xAstrom; }
    double  getXAstromErr() const { return _xAstromErr; }
    double  getYAstrom() const { return _yAstrom; }
    double  getYAstromErr() const { return _yAstromErr; }
    double  getRaAstrom() const { return _raAstrom; }
    double  getRaAstromErr() const { return _raAstromErr; }
    double  getDecAstrom() const { return _decAstrom; }
    double  getDecAstromErr() const { return _decAstromErr; }
    double  getTaiMidPoint() const { return _taiMidPoint; }
    float   getTaiRange() const { return _taiRange; }
    float   getFwhmA() const { return _fwhmA; }
    float   getFwhmB() const { return _fwhmB; }
    float   getFwhmTheta() const { return _fwhmTheta; }
    double  getPsfMag() const { return _psfMag; }
    float   getPsfMagErr() const { return _psfMagErr; }
    double  getApMag() const { return _apMag; }
    float   getApMagErr() const { return _apMagErr; }
    double  getModelMag() const { return _modelMag; }
    float   getModelMagErr() const { return _modelMagErr; }
    double  getInstMag() const { return _instMag; }
    double  getInstMagErr() const { return _instMagErr; }
    double  getNonGrayCorrMag() const { return _nonGrayCorrMag; }
    double  getNonGrayCorrMagErr() const { return _nonGrayCorrMagErr;}
    double  getAtmCorrMag() const { return _atmCorrMag; }
    double  getAtmCorrMagErr() const { return _atmCorrMagErr; }
    float   getApDia() const { return _apDia; }
    float   getSnr() const { return _snr; }
    float   getChi2() const { return _chi2; }
    boost::int16_t getFlagForAssociation() const { return _flagForAssociation; }
    boost::int16_t getFlagForDetection() const { return _flagForDetection; }
    boost::int16_t getFlagForWcs() const { return _flagForWcs; }
        
    // setters
    void setId(boost::int64_t const id) {
        set(_id, id);         
    }
    void setAmpExposureId(boost::int64_t const ampExposureId) {
        set(_ampExposureId, ampExposureId);
    }
    void setFilterId(boost::int8_t const filterId) {
        set(_filterId, filterId);         
    }
    void setObjectId(boost::int64_t const objectId) {
        set(_objectId, objectId, OBJECT_ID);
    }
    void setMovingObjectId(boost::int64_t const movingObjectId) {
        set(_movingObjectId, movingObjectId, MOVING_OBJECT_ID);
    }
    void setProcHistoryId(boost::int32_t const procHistoryId) {
        set(_procHistoryId, procHistoryId);   
    }   
    void setRa(double const ra) {
        set(_ra, ra);
    }
    void setDec(double const dec) {
        set(_dec, dec);
    }
    void setRaErrForWcs(float const raErrForWcs) {
        set(_raErrForWcs, raErrForWcs);
    }
    void setDecErrForWcs(float const decErrForWcs) {
        set(_decErrForWcs, decErrForWcs);
    }
    void setRaErrForDetection(float const raErrForDetection ) {
        set(_raErrForDetection, raErrForDetection);
    }
    void setDecErrForDetection(float const decErrForDetection) {
        set(_decErrForDetection, decErrForDetection);
    }
    void setXFlux(double const xFlux) { 
        set(_xFlux, xFlux, X_FLUX);            
    }
    void setXFluxErr(double const xFluxErr) { 
        set(_xFluxErr, xFluxErr, X_FLUX_ERR);            
    }    
    void setYFlux(double const yFlux) { 
        set(_yFlux, yFlux, Y_FLUX);            
    }    
    void setYFluxErr(double const yFluxErr) { 
        set(_yFluxErr, yFluxErr, Y_FLUX_ERR);            
    }    
    void setRaFlux(double const raFlux) { 
        set(_raFlux, raFlux, RA_FLUX);            
    }
    void setRaFluxErr(double const raFluxErr) { 
        set(_raFluxErr, raFluxErr, RA_FLUX_ERR);            
    }    
    void setDecFlux(double const decFlux) { 
        set(_decFlux, decFlux, DEC_FLUX);
    }    
    void setDecFluxErr(double const decFluxErr) { 
        set(_decFluxErr, decFluxErr, DEC_FLUX_ERR);            
    }    
    void setXPeak(double const xPeak) { 
        set(_xPeak, xPeak, X_PEAK);            
    }
    void setYPeak(double const yPeak) { 
        set(_yPeak, yPeak, Y_PEAK);            
    }    
    void setRaPeak(double const raPeak) { 
        set(_raPeak, raPeak, RA_PEAK);            
    }    
    void setDecPeak(double const decPeak) { 
        set(_decPeak, decPeak, DEC_PEAK);            
    }    
    void setXAstrom(double const xAstrom) { 
        set(_xAstrom, xAstrom, X_ASTROM);            
    }
    void setXAstromErr(double const xAstromErr) { 
        set(_xAstromErr, xAstromErr, X_ASTROM_ERR);            
    }    
    void setYAstrom(double const yAstrom) { 
        set(_yAstrom, yAstrom, Y_ASTROM);            
    }    
    void setYAstromErr(double const yAstromErr) { 
        set(_yAstromErr, yAstromErr, Y_ASTROM_ERR);            
    }    
    void setRaAstrom(double const raAstrom) { 
        set(_raAstrom, raAstrom, RA_ASTROM);            
    }
    void setRaAstromErr(double const raAstromErr) { 
        set(_raAstromErr, raAstromErr, RA_ASTROM_ERR);            
    }    
    void setDecAstrom(double const decAstrom) { 
        set(_decAstrom, decAstrom, DEC_ASTROM);            
    }    
    void setDecAstromErr(double const decAstromErr) { 
        set(_decAstromErr, decAstromErr, DEC_ASTROM_ERR);            
    }         
    void setTaiMidPoint(double const taiMidPoint) {
        set(_taiMidPoint, taiMidPoint);     
    }
    void setTaiRange(float const taiRange) {
        set(_taiRange, taiRange);         
    }
    void setFwhmA(float const fwhmA) {
        set(_fwhmA, fwhmA);
    }
    void setFwhmB(float const fwhmB) {
        set(_fwhmB, fwhmB);
    }
    void setFwhmTheta(float const fwhmTheta) {
        set(_fwhmTheta, fwhmTheta);
    }
    void setPsfMag(double const psfMag) {
        set(_psfMag, psfMag);           
    }
    void setPsfMagErr(float const psfMagErr) {
        set(_psfMagErr, psfMagErr);       
    }
    void setApMag(double const apMag) {
        set(_apMag, apMag);           
    }
    void setApMagErr(float const apMagErr) {
        set(_apMagErr, apMagErr);         
    }
    void setModelMag(double const modelMag) {
        set(_modelMag, modelMag);
    }
    void setModelMagErr(float const modelMagErr) {
        set(_modelMagErr, modelMagErr);
    }
    void setInstMag(double const instMag) {
        set(_instMag, instMag);
    }
    void setInstMagErr(double const instMagErr      ) {
        set(_instMagErr, instMagErr);     
    }
    void setNonGrayCorrMag(double const nonGrayCorrMag) { 
        set(_nonGrayCorrMag, nonGrayCorrMag,  NON_GRAY_CORR_MAG);         
    }
    void setNonGrayCorrMagErr(double const nonGrayCorrMagErr) { 
        set(_nonGrayCorrMagErr, nonGrayCorrMagErr,  NON_GRAY_CORR_MAG_ERR);      
    }
    void setAtmCorrMag(double const atmCorrMag) { 
        set(_atmCorrMag, atmCorrMag, ATM_CORR_MAG);         
    }
    void setAtmCorrMagErr(double const atmCorrMagErr) { 
        set(_atmCorrMagErr, atmCorrMagErr, ATM_CORR_MAG_ERR);      
    }     
    void setApDia(float const apDia) {
        set(_apDia, apDia, AP_DIA);
    }
    void setSnr(float const snr) {
        set(_snr, snr);             
    }
    void setChi2(float const chi2) {
        set(_chi2, chi2);             
    }   
    void setFlagForAssociation(boost::int16_t const flagForAssociation) {
        set(_flagForAssociation, flagForAssociation, FLAG_FOR_ASSOCIATION);
    }
    void setFlagForDetection(boost::int16_t const flagForDetection) {
        set(_flagForDetection, flagForDetection, FLAG_FOR_DETECTION);
    }
    void setFlagForWcs(boost::int16_t const flagForWcs) {
        set(_flagForWcs, flagForWcs, FLAG_FOR_WCS);
    }    
    
    /**
     * Test if a field is Null
     */
    inline bool isNull(int const field) const { 
        if (field >= 0 && field < numNullableFields)        
            return _nulls.test(field); 
        else return false;
    }

    /**
     * Set field to NOT NULL
     */
    inline void setNotNull(int const field) {
        if (field >= 0 && field < numNullableFields)
            _nulls.reset(field);
    }

    /**
     * Set field to null
     */    
    inline void setNull(int const field, bool const null = true) { 
        if (field >= 0 && field < numNullableFields)
            _nulls.set(field, null);   
    }
    
    /**
     * Set all NULLABLE fields to NULL
     */
    inline void setNull() { _nulls.set();}
    
    /**
     * Set all NULLABLE fields to NOT NULL
     */
    inline void setNotNull() {_nulls.reset();}   
protected:
    
    std::bitset<numNullableFields> _nulls;
    
    /**
     * Default Constructor
     */
    BaseSourceAttributes(): 
        _id(0), _ampExposureId(0), 
        _objectId(0), _movingObjectId(0),                 
        _ra(0.0), _dec(0.0), 
        _xFlux(0.0), _xFluxErr(0.0),
        _yFlux(0.0), _yFluxErr(0.0),
        _raFlux(0.0),_raFluxErr(0.0),
        _decFlux(0.0), _decFluxErr(0.0),
        _xPeak(0.0), _yPeak(0.0), _raPeak(0.0), _decPeak(0.0),
        _xAstrom(0.0), _xAstromErr(0.0),
        _yAstrom(0.0), _yAstromErr(0.0),
        _raAstrom(0.0), _raAstromErr(0.0),
        _decAstrom(0.0), _decAstromErr(0.0),
        _taiMidPoint(0.0), 
        _psfMag(0.0), _apMag(0.0), _modelMag(0.0),
        _instMag(0.0), _instMagErr(0.0),
        _nonGrayCorrMag(0.0), _nonGrayCorrMagErr(0.0),
        _atmCorrMag(0.0), _atmCorrMagErr(0.0),
        _raErrForDetection(0.0), _decErrForDetection(0.0),
        _raErrForWcs(0.0), _decErrForWcs(0.0),                
        _taiRange(0.0),
        _fwhmA(0.0), _fwhmB(0.0), _fwhmTheta(0.0),
        _psfMagErr(0.0), _apMagErr(0.0), _modelMagErr(0.0),
        _apDia(0.0), _snr(0.0), _chi2(0.0),
        _procHistoryId(0),
        _flagForAssociation(0), _flagForDetection(0), _flagForWcs(0),
        _filterId(0)
    {
        setNull();
    }    
    
    /**
     * \internal test field equality
     * \return true if equal or null
     */
    template<typename T> 
    inline bool areEqual(T const & a, T const & b, int const field = -1) const {
    
        bool null = isNull(field);
        
        return (a == b || null);
    }

    /**
     * \internal Set the value of a field, and if it is null
     */    
    template<typename T>
    inline void set(T &dest, T const & src, int const field = -1) {
        setNotNull(field);            
        dest = src;
    }
    
 
    /**
     * \internal Serialize field values, and null statuses          
     */    
    template <class Archive> 
    void serialize(Archive & ar, unsigned int const version) {
        ar & _id;
        ar & _ampExposureId;
        ar & _filterId;
        ar & _objectId;
        ar & _movingObjectId;
        ar & _procHistoryId;
        ar & _ra;
        ar & _dec;
        ar & _raErrForDetection;
        ar & _decErrForDetection;
        ar & _raErrForWcs;
        ar & _decErrForWcs;
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
        ar & _flagForAssociation;
        ar & _flagForDetection;
        ar & _flagForWcs;
        
        bool b;
        if (Archive::is_loading::value) {
            for (int i = 0; i != numNullableFields; ++i) {
                ar & b;
                _nulls.set(i, b);
            }
        } else {
            for (int i = 0; i != numNullableFields; ++i) {
                b = isNull(i);
                ar & b;
            }
        }    
    }
            
    
    boost::int64_t _id;
    boost::int64_t _ampExposureId;
    boost::int64_t _objectId;
    boost::int64_t _movingObjectId;
    double  _ra;
    double  _dec;
    double  _xFlux;
    double  _xFluxErr;
    double  _yFlux;
    double  _yFluxErr;
    double  _raFlux;
    double  _raFluxErr;
    double  _decFlux;
    double  _decFluxErr;
    double  _xPeak;
    double  _yPeak;
    double  _raPeak;
    double  _decPeak;
    double  _xAstrom;
    double  _xAstromErr;
    double  _yAstrom;
    double  _yAstromErr;
    double  _raAstrom;
    double  _raAstromErr;
    double  _decAstrom;
    double  _decAstromErr;
    double  _taiMidPoint;
    double  _psfMag;
    double  _apMag;
    double  _modelMag;
    double  _instMag;
    double  _instMagErr;
    double  _nonGrayCorrMag;
    double  _nonGrayCorrMagErr;
    double  _atmCorrMag;
    double  _atmCorrMagErr;
    float   _raErrForDetection;
    float   _decErrForDetection;
    float   _raErrForWcs;
    float   _decErrForWcs;
    float   _taiRange;
    float   _fwhmA;
    float   _fwhmB;
    float   _fwhmTheta;
    float   _psfMagErr;
    float   _apMagErr;
    float   _modelMagErr;
    float   _apDia;
    float   _snr;
    float   _chi2;
    boost::int32_t _procHistoryId;
    boost::int16_t _flagForAssociation;
    boost::int16_t _flagForDetection;
    boost::int16_t _flagForWcs;
    boost::int8_t  _filterId;
    
    friend class boost::serialization::access;
};


}}} //namespace lsst::afw::detection

#endif
