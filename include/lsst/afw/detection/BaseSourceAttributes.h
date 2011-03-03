/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
#ifndef LSST_AFW_DETECTION_BASE_SOURCE_ATTRIBUTES_H
#define LSST_AFW_DETECTION_BASE_SOURCE_ATTRIBUTES_H

#include <math.h>

#include <bitset>
#include <limits>
#include "boost/cstdint.hpp"
#include "lsst/utils/ieee.h"

#include "lsst/afw/image/Filter.h"


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
    RA_ERR_FOR_DETECTION,
    DEC_ERR_FOR_DETECTION,
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
    X_ASTROM_ERR,
    Y_ASTROM_ERR,    
    RA_ASTROM,
    RA_ASTROM_ERR,
    DEC_ASTROM,
    DEC_ASTROM_ERR,
    NON_GRAY_CORR_FLUX,
    NON_GRAY_CORR_FLUX_ERR,
    ATM_CORR_FLUX,        
    ATM_CORR_FLUX_ERR,        
    AP_DIA,      
    IXX,
    IXX_ERR,
    IYY,
    IYY_ERR,
    IXY,
    IXY_ERR,
    PSF_IXX,
    PSF_IXX_ERR,
    PSF_IYY,
    PSF_IYY_ERR,
    PSF_IXY,
    PSF_IXY_ERR,
    E1,
    E1_ERR,
    E2,
    E2_ERR,
    RESOLUTION,
    SHEAR1,
    SHEAR1_ERR,
    SHEAR2,
    SHEAR2_ERR,
    SIGMA,
    SIGMA_ERR,
    SHAPE_STATUS,
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
    double getRa() const { return _ra; }
    double getDec() const { return _dec; }
    float  getRaErrForWcs() const { return _raErrForWcs; }
    float  getDecErrForWcs() const { return _decErrForWcs; }
    float  getRaErrForDetection() const { return _raErrForDetection; }
    float  getDecErrForDetection() const { return _decErrForDetection; }
    double getXFlux() const { return _xFlux; }
    float  getXFluxErr() const { return _xFluxErr; }
    double getYFlux() const { return _yFlux; }
    float  getYFluxErr() const { return _yFluxErr; }
    double getRaFlux() const { return _raFlux; }
    float  getRaFluxErr() const { return _raFluxErr; }
    double getDecFlux() const { return _decFlux; }
    float  getDecFluxErr() const { return _decFluxErr; }
    double getXPeak() const { return _xPeak; }
    double getYPeak() const { return _yPeak; }
    double getRaPeak() const { return _raPeak; }
    double getDecPeak() const { return _decPeak; }
    double getXAstrom() const { return _xAstrom; }
    float  getXAstromErr() const { return _xAstromErr; }
    double getYAstrom() const { return _yAstrom; }
    float  getYAstromErr() const { return _yAstromErr; }
    double getRaAstrom() const { return _raAstrom; }
    float  getRaAstromErr() const { return _raAstromErr; }
    double getDecAstrom() const { return _decAstrom; }
    float  getDecAstromErr() const { return _decAstromErr; }
    double getTaiMidPoint() const { return _taiMidPoint; }
    double getTaiRange() const { return _taiRange; }
    double getPsfFlux() const { return _psfFlux; }
    float  getPsfFluxErr() const { return _psfFluxErr; }
    double getApFlux() const { return _apFlux; }
    float  getApFluxErr() const { return _apFluxErr; }
    double getModelFlux() const { return _modelFlux; }
    float  getModelFluxErr() const { return _modelFluxErr; }
    double getInstFlux() const { return _instFlux; }
    float  getInstFluxErr() const { return _instFluxErr; }      
    double getNonGrayCorrFlux() const { return _nonGrayCorrFlux; }
    float  getNonGrayCorrFluxErr() const { return _nonGrayCorrFluxErr;}
    double getAtmCorrFlux() const { return _atmCorrFlux; }
    float  getAtmCorrFluxErr() const { return _atmCorrFluxErr; }
    float  getApDia() const { return _apDia; }
    float  getIxx() const { return _ixx; }
    float  getIxxErr() const { return _ixxErr; }
    float  getIyy() const { return _iyy; }
    float  getIyyErr() const { return _iyyErr; }
    float  getIxy() const { return _ixy; }
    float  getIxyErr() const { return _ixyErr; }

    float  getPsfIxx() const { return _psfIxx; }
    float  getPsfIxxErr() const { return _psfIxxErr; }
    float  getPsfIyy() const { return _psfIyy; }
    float  getPsfIyyErr() const { return _psfIyyErr; }
    float  getPsfIxy() const { return _psfIxy; }
    float  getPsfIxyErr() const { return _psfIxyErr; }

    float  getResolution() const { return _resolution; }

    float  getE1() const { return _e1; }
    float  getE1Err() const { return _e1Err; }
    float  getE2() const { return _e2; }
    float  getE2Err() const { return _e2Err; }
    float  getShear1() const { return _shear1; }
    float  getShear1Err() const { return _shear1Err; }
    float  getShear2() const { return _shear2; }
    float  getShear2Err() const { return _shear2Err; }

    float  getSigma() const { return _sigma; }
    float  getSigmaErr() const { return _sigmaErr; }
    boost::int16_t getShapeStatus() const { return _shapeStatus; }
    
    float  getSnr() const { return _snr; }
    float  getChi2() const { return _chi2; }
    boost::int16_t getFlagForAssociation() const { return _flagForAssociation; }
    boost::int64_t getFlagForDetection() const { return _flagForDetection; }
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
        set(_raErrForDetection, raErrForDetection, RA_ERR_FOR_DETECTION);
    }
    void setDecErrForDetection(float const decErrForDetection) {
        set(_decErrForDetection, decErrForDetection, DEC_ERR_FOR_DETECTION);
    }
    void setXFlux(double const xFlux) { 
        set(_xFlux, xFlux, X_FLUX);            
    }
    void setXFluxErr(float const xFluxErr) { 
        set(_xFluxErr, xFluxErr, X_FLUX_ERR);            
    }    
    void setYFlux(double const yFlux) { 
        set(_yFlux, yFlux, Y_FLUX);            
    }    
    void setYFluxErr(float const yFluxErr) { 
        set(_yFluxErr, yFluxErr, Y_FLUX_ERR);            
    }    
    void setRaFlux(double const raFlux) { 
        set(_raFlux, raFlux, RA_FLUX);            
    }
    void setRaFluxErr(float const raFluxErr) { 
        set(_raFluxErr, raFluxErr, RA_FLUX_ERR);            
    }    
    void setDecFlux(double const decFlux) { 
        set(_decFlux, decFlux, DEC_FLUX);
    }    
    void setDecFluxErr(float const decFluxErr) { 
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
        set(_xAstrom, xAstrom);            
    }
    void setXAstromErr(float const xAstromErr) { 
        set(_xAstromErr, xAstromErr, X_ASTROM_ERR);            
    }    
    void setYAstrom(double const yAstrom) { 
        set(_yAstrom, yAstrom);            
    }    
    void setYAstromErr(float const yAstromErr) {     
        set(_yAstromErr, yAstromErr, Y_ASTROM_ERR);            
    }    
    void setRaAstrom(double const raAstrom) { 
        set(_raAstrom, raAstrom, RA_ASTROM);            
    }
    void setRaAstromErr(float const raAstromErr) { 
        set(_raAstromErr, raAstromErr, RA_ASTROM_ERR);            
    }    
    void setDecAstrom(double const decAstrom) { 
        set(_decAstrom, decAstrom, DEC_ASTROM);            
    }    
    void setDecAstromErr(float const decAstromErr) { 
        set(_decAstromErr, decAstromErr, DEC_ASTROM_ERR);            
    }         
    void setTaiMidPoint(double const taiMidPoint) {
        set(_taiMidPoint, taiMidPoint);     
    }
    void setTaiRange(double const taiRange) {
        set(_taiRange, taiRange);         
    }

    void setPsfFlux(double const psfFlux) {
        set(_psfFlux, psfFlux);           
    }
    void setPsfFluxErr(float const psfFluxErr) {
        set(_psfFluxErr, psfFluxErr);       
    }
    void setApFlux(double const apFlux) {
        set(_apFlux, apFlux);           
    }
    void setApFluxErr(float const apFluxErr) {
        set(_apFluxErr, apFluxErr);         
    }
    void setModelFlux(double const modelFlux) {
        set(_modelFlux, modelFlux);
    }
    void setModelFluxErr(float const modelFluxErr) {
        set(_modelFluxErr, modelFluxErr);
    }
    void setInstFlux(double const instFlux) {
        set(_instFlux, instFlux);
    }
    void setInstFluxErr(float const instFluxErr      ) {
        set(_instFluxErr, instFluxErr);     
    }
    void setNonGrayCorrFlux(double const nonGrayCorrFlux) { 
        set(_nonGrayCorrFlux, nonGrayCorrFlux,  NON_GRAY_CORR_FLUX);         
    }
    void setNonGrayCorrFluxErr(float const nonGrayCorrFluxErr) {        
        set(_nonGrayCorrFluxErr, nonGrayCorrFluxErr,  NON_GRAY_CORR_FLUX_ERR);      
    }
    void setAtmCorrFlux(double const atmCorrFlux) { 
        set(_atmCorrFlux, atmCorrFlux, ATM_CORR_FLUX);         
    }
    void setAtmCorrFluxErr(float const atmCorrFluxErr) { 
        set(_atmCorrFluxErr, atmCorrFluxErr, ATM_CORR_FLUX_ERR);      
    }     
    void setIxx(float const ixx) { 
        set(_ixx, ixx, IXX);    
    }
    void setIxxErr(float const ixxErr) {
        set(_ixxErr, ixxErr, IXX_ERR); 
    }         
    void setIyy(float const iyy) { 
        set(_iyy, iyy, IYY);    
    }     
    void setIyyErr(float const iyyErr) { 
        set(_iyyErr, iyyErr, IYY_ERR); 
    }         
    void setIxy(float const ixy) { 
        set(_ixy, ixy, IXY);    
    }      
    void setIxyErr(float const ixyErr) { 
        set(_ixyErr, ixyErr, IXY_ERR); 
    }

    void setPsfIxx(float const psfIxx) { 
        set(_psfIxx, psfIxx, PSF_IXX);    
    }
    void setPsfIxxErr(float const psfIxxErr) {
        set(_psfIxxErr, psfIxxErr, PSF_IXX_ERR); 
    }         
    void setPsfIyy(float const psfIyy) { 
        set(_psfIyy, psfIyy, PSF_IYY);    
    }     
    void setPsfIyyErr(float const psfIyyErr) { 
        set(_psfIyyErr, psfIyyErr, PSF_IYY_ERR); 
    }         
    void setPsfIxy(float const psfIxy) { 
        set(_psfIxy, psfIxy, PSF_IXY);    
    }      
    void setPsfIxyErr(float const psfIxyErr) { 
        set(_psfIxyErr, psfIxyErr, PSF_IXY_ERR); 
    }         

    void setE1(float const e1) {
        set(_e1, e1, E1);    
    }      
    void setE1Err(float const e1Err) {
        set(_e1Err, e1Err, E1_ERR);    
    }      
    void setE2(float const e2) {
        set(_e2, e2, E2);    
    }      
    void setE2Err(float const e2Err) {
        set(_e2Err, e2Err, E2_ERR);    
    }      
    void setShear1(float const shear1) {
        set(_shear1, shear1, SHEAR1);    
    }      
    void setShear1Err(float const shear1Err) {
        set(_shear1Err, shear1Err, SHEAR1_ERR);    
    }      
    void setShear2(float const shear2) {
        set(_shear2, shear2, SHEAR2);    
    }      
    void setShear2Err(float const shear2Err) {
        set(_shear2Err, shear2Err, SHEAR2_ERR);    
    }      

    void setResolution(float const resolution) {
        set(_resolution, resolution, RESOLUTION);
    }

    void setSigma(float const sigma) {
        set(_sigma, sigma, SIGMA);
    }
    void setSigmaErr(float const sigmaErr) {
        set(_sigmaErr, sigmaErr, SIGMA_ERR);
    }
    void setShapeStatus(boost::int16_t const status) {
        set(_shapeStatus, status, SHAPE_STATUS);
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
    void setFlagForDetection(boost::int64_t const flagForDetection) {
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
    BaseSourceAttributes(boost::int64_t id=0): 
        _id(id), _ampExposureId(0), 
        _objectId(0), _movingObjectId(0),                 
        _flagForDetection(0), 
        _ra(0.0), _dec(0.0), 
        _xFlux(0.0), _yFlux(0.0),
        _raFlux(0.0), _decFlux(0.0),
        _xPeak(0.0), _yPeak(0.0),
        _raPeak(0.0), _decPeak(0.0),
        _xAstrom(0.0), _yAstrom(0.0),
        _raAstrom(0.0), _decAstrom(0.0), 
        _taiMidPoint(0.0), _taiRange(0.0), 
        _psfFlux(0.0), _apFlux(0.0), _modelFlux(0.0),
        _instFlux(0.0),
        _nonGrayCorrFlux(0.0), _atmCorrFlux(0.0),
        _raErrForDetection(0.0), _decErrForDetection(0.0),
        _raErrForWcs(0.0), _decErrForWcs(0.0),                
        _xFluxErr(0.0), _yFluxErr(0.0),
        _raFluxErr(0.0), _decFluxErr(0.0), 
        _xAstromErr(0.0), _yAstromErr(0.0),
        _raAstromErr(0.0), _decAstromErr(0.0),
        _psfFluxErr(0.0),
        _apFluxErr(0.0),
        _modelFluxErr(0.0),
        _instFluxErr(0.0),
        _nonGrayCorrFluxErr(0.0),
        _atmCorrFluxErr(0.0),
        _apDia(0.0), 
        _ixx(0.0), _ixxErr(0.0),
        _iyy(0.0), _iyyErr(0.0),
        _ixy(0.0), _ixyErr(0.0),    
        _psfIxx(0.0), _psfIxxErr(0.0),
        _psfIyy(0.0), _psfIyyErr(0.0),
        _psfIxy(0.0), _psfIxyErr(0.0),
        _resolution(0.0),
        _sigma(0.0), _sigmaErr(0.0),
        _e1(0.0), _e1Err(0.0),
        _e2(0.0), _e2Err(0.0),
        _shear1(0.0), _shear1Err(0.0),
        _shear2(0.0), _shear2Err(0.0),
        _snr(0.0), _chi2(0.0),
        _procHistoryId(0),
        _flagForAssociation(0), _flagForWcs(0),
        _filterId(lsst::afw::image::Filter::UNKNOWN),
        _shapeStatus(-1)
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
    inline bool areEqual(float const & a, float const & b, int const field = -1) const {
        bool null = isNull(field);
        return (lsst::utils::isnan(a) ? lsst::utils::isnan(b) : a == b) || null;
    }
    inline bool areEqual(double const & a, double const & b, int const field = -1) const {
        bool null = isNull(field);
        return (lsst::utils::isnan(a) ? lsst::utils::isnan(b) : a == b) || null;
    }

    /**
     * \internal Set the value of a field, and if it is null
     */
    template<typename T>
    inline void set(T & dest, T const & src, int const field = -1) {
        setNotNull(field);
        dest = src;
    }

    template <typename Archive, typename FloatT>
    static inline void fpSerialize(Archive & ar, FloatT & value) {
        int fpClass = 0;
        if (lsst::utils::isnan(value)) {
            fpClass = 1;
        } else if (lsst::utils::isinf(value)) {
            fpClass = value > 0.0 ? 2 : 3;
        }
        ar & fpClass;
        switch (fpClass) {
            case 1:
                value = std::numeric_limits<FloatT>::quiet_NaN();
                break;
            case 2:
                value = std::numeric_limits<FloatT>::infinity();
                break;
            case 3:
                value = -std::numeric_limits<FloatT>::infinity();
                break;
            default:
                ar & value;
        }
    }

    /**
     * \internal Serialize field values, and null statuses
     */
    template <class Archive> 
    void serialize(Archive & ar, unsigned int const) {
        ar & _id;
        ar & _ampExposureId;
        ar & _filterId;
        ar & _objectId;
        ar & _movingObjectId;
        ar & _procHistoryId;
        fpSerialize(ar, _ra);
        fpSerialize(ar, _dec);
        fpSerialize(ar, _raErrForDetection);
        fpSerialize(ar, _decErrForDetection);
        fpSerialize(ar, _raErrForWcs);
        fpSerialize(ar, _decErrForWcs);
        fpSerialize(ar, _xFlux);
        fpSerialize(ar, _xFluxErr);
        fpSerialize(ar, _yFlux);
        fpSerialize(ar, _yFluxErr);
        fpSerialize(ar, _raFlux);
        fpSerialize(ar, _raFluxErr);
        fpSerialize(ar, _decFlux);
        fpSerialize(ar, _decFluxErr);
        fpSerialize(ar, _xPeak);
        fpSerialize(ar, _yPeak);
        fpSerialize(ar, _raPeak);
        fpSerialize(ar, _decPeak);
        fpSerialize(ar, _xAstrom);
        fpSerialize(ar, _xAstromErr);
        fpSerialize(ar, _yAstrom);
        fpSerialize(ar, _yAstromErr);
        fpSerialize(ar, _raAstrom);
        fpSerialize(ar, _raAstromErr);
        fpSerialize(ar, _decAstrom);
        fpSerialize(ar, _decAstromErr);
        fpSerialize(ar, _taiMidPoint);
        fpSerialize(ar, _taiRange);
        fpSerialize(ar, _psfFlux);
        fpSerialize(ar, _psfFluxErr);
        fpSerialize(ar, _apFlux);
        fpSerialize(ar, _apFluxErr);
        fpSerialize(ar, _modelFlux);
        fpSerialize(ar, _modelFluxErr);
        fpSerialize(ar, _instFlux);
        fpSerialize(ar, _instFluxErr);
        fpSerialize(ar, _nonGrayCorrFlux);
        fpSerialize(ar, _nonGrayCorrFluxErr);
        fpSerialize(ar, _atmCorrFlux);
        fpSerialize(ar, _atmCorrFluxErr);
        fpSerialize(ar, _apDia);
        fpSerialize(ar, _ixx);
        fpSerialize(ar, _ixxErr);
        fpSerialize(ar, _iyy);
        fpSerialize(ar, _iyyErr);
        fpSerialize(ar, _ixy);
        fpSerialize(ar, _ixyErr);
        fpSerialize(ar, _psfIxx);
        fpSerialize(ar, _psfIxxErr);
        fpSerialize(ar, _psfIyy);
        fpSerialize(ar, _psfIyyErr);
        fpSerialize(ar, _psfIxy);
        fpSerialize(ar, _psfIxyErr);
        fpSerialize(ar, _resolution);
        fpSerialize(ar, _sigma);
        fpSerialize(ar, _sigmaErr);
        fpSerialize(ar, _shapeStatus);
        fpSerialize(ar, _e1);
        fpSerialize(ar, _e1Err);
        fpSerialize(ar, _e2);
        fpSerialize(ar, _e2Err);
        fpSerialize(ar, _shear1);
        fpSerialize(ar, _shear1Err);
        fpSerialize(ar, _shear2);
        fpSerialize(ar, _shear2Err);
        fpSerialize(ar, _snr);
        fpSerialize(ar, _chi2);
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
    boost::int64_t _flagForDetection;
    double _ra;
    double _dec;
    double _xFlux;
    double _yFlux;
    double _raFlux;
    double _decFlux;
    double _xPeak;
    double _yPeak;
    double _raPeak;
    double _decPeak;
    double _xAstrom;
    double _yAstrom;
    double _raAstrom;
    double _decAstrom;
    double _taiMidPoint;
    double _taiRange;
    double _psfFlux;
    double _apFlux;
    double _modelFlux;
    double _instFlux;
    double _nonGrayCorrFlux;
    double _atmCorrFlux;
    float _raErrForDetection;
    float _decErrForDetection;
    float _raErrForWcs;
    float _decErrForWcs;
    float _xFluxErr;
    float _yFluxErr;
    float _raFluxErr;
    float _decFluxErr;
    float _xAstromErr;
    float _yAstromErr;
    float _raAstromErr;
    float _decAstromErr;
    float _psfFluxErr;
    float _apFluxErr;
    float _modelFluxErr;
    float _instFluxErr;
    float _nonGrayCorrFluxErr;
    float _atmCorrFluxErr;
    float _apDia;
    float _ixx;
    float _ixxErr;
    float _iyy;
    float _iyyErr;
    float _ixy;
    float _ixyErr;
    float _psfIxx;
    float _psfIxxErr;
    float _psfIyy;
    float _psfIyyErr;
    float _psfIxy;
    float _psfIxyErr;
    float _resolution;
    float _sigma;
    float _sigmaErr;
    float _e1;
    float _e1Err;
    float _e2;
    float _e2Err;
    float _shear1;
    float _shear1Err;
    float _shear2;
    float _shear2Err;
    float _snr;
    float _chi2;
    boost::int32_t _procHistoryId;
    boost::int16_t _flagForAssociation;
    boost::int16_t _flagForWcs;
    boost::int8_t _filterId;
    boost::int16_t _shapeStatus;
    
    friend class boost::serialization::access;
};


}}} //namespace lsst::afw::detection

#endif
