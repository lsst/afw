// -*- lsst-c++ -*-

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
 
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  The C++ representation of a Deep Detection Source.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_DETECTION_SOURCE_H
#define LSST_AFW_DETECTION_SOURCE_H

#include <bitset>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include "boost/serialization/shared_ptr.hpp"

#include "lsst/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/detection/BaseSourceAttributes.h"

/*
 * Avoid bug in lsst/base.h; it's fixed in base 3.1.3 
 */
#undef PTR
#define LSST_WHITESPACE /* White space to avoid swig converting vector<PTR(XX)> into vector<shared_ptr<XX>> */
#define PTR(...) boost::shared_ptr<__VA_ARGS__ LSST_WHITESPACE > LSST_WHITESPACE

namespace boost {
namespace serialization {
    class access;
}}

namespace lsst {
namespace afw {
    namespace formatters {
        class SourceVectorFormatter;
    }
    
namespace detection {
    class Footprint;
    template<typename T> class Measurement;
    class Astrometry;
    class Photometry;
    class Shape;
    
/*! An integer id for each nullable field in Source. */
enum SourceNullableField {
    AMP_EXPOSURE_ID = NUM_SHARED_NULLABLE_FIELDS,
    TAI_RANGE,
    X_ASTROM,
    Y_ASTROM,
    PETRO_FLUX,
    PETRO_FLUX_ERR,
    SKY,
    SKY_ERR,
    RA_OBJECT,
    DEC_OBJECT,
    NUM_SOURCE_NULLABLE_FIELDS
};


/**
 * In-code representation of an entry in the Source catalog for
 *   persisting/retrieving Sources
 */
class Source 
    : public BaseSourceAttributes< NUM_SOURCE_NULLABLE_FIELDS> {
public :
    typedef boost::shared_ptr<Source> Ptr;

    Source(int id=0, PTR(Footprint)=PTR(Footprint)());
    Source(Source const & other);  
    virtual ~Source(){};

    // getters
    boost::int64_t getSourceId() const { return _id; }
    double getPetroFlux() const { return _petroFlux; }
    float  getPetroFluxErr() const { return _petroFluxErr; }    
    float  getSky() const { return _sky; }
    float  getSkyErr() const { return _skyErr; }
    // in radians
    double getRaObject() const { return _raObject; }
    // in radians
    double getDecObject() const { return _decObject; }

#ifndef SWIG
    CONST_PTR(Footprint) getFootprint() const { return _footprint; }
    void setFootprint(CONST_PTR(Footprint) footprint) { _footprint = footprint; }
#endif

    // setters
    void setSourceId( boost::int64_t const sourceId) {setId(sourceId);}


    void setPetroFlux(double const petroFlux) { 
        set(_petroFlux, petroFlux, PETRO_FLUX);         
    }
    void setPetroFluxErr(float const petroFluxErr) { 
        set(_petroFluxErr, petroFluxErr, PETRO_FLUX_ERR);    
    }
    void setSky(float const sky) { 
        set(_sky, sky, SKY);       
    }
    void setSkyErr (float const skyErr) {
        set(_skyErr, skyErr, SKY_ERR);
    }
    // in radians
    void setRaObject(double const raObject) {
        set(_raObject, raObject, RA_OBJECT);
    }
    // in radians
    void setDecObject(double const decObject) {
        set(_decObject, decObject, DEC_OBJECT);
    }

    void setRaDecObject(lsst::afw::coord::Coord::ConstPtr radec) {
        // Convert to LSST-decreed ICRS and radians.
        lsst::afw::coord::IcrsCoord icrs = radec->toIcrs();
        setRaObject(icrs.getRa(lsst::afw::coord::RADIANS));
        setDecObject(icrs.getDec(lsst::afw::coord::RADIANS));
    }

    void setAllRaDecFields(lsst::afw::coord::Coord::ConstPtr radec) {
        setRaDecObject(radec);
        BaseSourceAttributes<NUM_SOURCE_NULLABLE_FIELDS>::setAllRaDecFields(radec);
    }

    //overloaded setters
    //Because these fields are not NULLABLE in all sources, 
    //  special behavior must be defined in the derived class
    void setAmpExposureId (boost::int64_t const ampExposureId) { 
        set(_ampExposureId, ampExposureId, AMP_EXPOSURE_ID);
    }
    void setTaiRange (double const taiRange) { 
        set(_taiRange, taiRange, TAI_RANGE);         
    }
    void setXAstrom(double const xAstrom) { 
        set(_xAstrom, xAstrom, X_ASTROM);            
    }
    void setYAstrom(double const yAstrom) { 
        set(_yAstrom, yAstrom, Y_ASTROM);            
    }
    void setAstrometry(PTR(lsst::afw::detection::Measurement<lsst::afw::detection::Astrometry>) astrom) {
        _astrom = astrom;
    }
    PTR(lsst::afw::detection::Measurement<lsst::afw::detection::Astrometry>) getAstrometry() const {
        return _astrom;
    }
    void setPhotometry(PTR(lsst::afw::detection::Measurement<lsst::afw::detection::Photometry>) photom) {
        _photom = photom;
    }
    PTR(lsst::afw::detection::Measurement<lsst::afw::detection::Photometry>) getPhotometry() const {
        return _photom;
    }
    void setShape(PTR(lsst::afw::detection::Measurement<lsst::afw::detection::Shape>) shape) {
        _shape = shape;
    }
    PTR(lsst::afw::detection::Measurement<lsst::afw::detection::Shape>) getShape() const {
        return _shape;
    }
    
    bool operator==(Source const & d) const;

private :
    CONST_PTR(Footprint) _footprint;

    double _raObject;
    double _decObject;
    double _petroFlux;  
    float _petroFluxErr;
    float _sky;
    float _skyErr;

    template <typename Archive> 
    void serialize(Archive & ar, unsigned int const version) {
        fpSerialize(ar, _raObject);
        fpSerialize(ar, _decObject);
        fpSerialize(ar, _petroFlux);
        fpSerialize(ar, _petroFluxErr);
        fpSerialize(ar, _sky);
        fpSerialize(ar, _skyErr);

        BaseSourceAttributes<NUM_SOURCE_NULLABLE_FIELDS>::serialize(ar, version);
        if (version > 0) {
            ar & make_nvp("astrom", _astrom) & make_nvp("photom", _photom) & make_nvp("shape", _shape);
        }
    }

    friend class boost::serialization::access;
    friend class formatters::SourceVectorFormatter;   

    PTR(Measurement<Astrometry>) _astrom;
    PTR(Measurement<Photometry>) _photom;
    PTR(Measurement<Shape>)      _shape;
};

inline bool operator!=(Source const & lhs, Source const & rhs) {
    return !(lhs==rhs);
}


typedef std::vector<Source::Ptr> SourceSet;
 
class PersistableSourceVector : public lsst::daf::base::Persistable {
public:
    typedef boost::shared_ptr<PersistableSourceVector> Ptr;
    PersistableSourceVector() {}
    PersistableSourceVector(SourceSet const & sources)
        : _sources(sources) {}
    ~PersistableSourceVector(){_sources.clear();}
        
    SourceSet getSources() const {return _sources; }
    void setSources(SourceSet const & sources) {_sources = sources; }
    
    bool operator==(SourceSet const & other) const {
        if (_sources.size() != other.size())
            return false;
                    
        SourceSet::size_type i;
        for (i = 0; i < _sources.size(); ++i) {
            if (*_sources[i] != *other[i])
                return false;            
        }
        
        return true;
    }
    
    bool operator==(PersistableSourceVector const & other) const {
        return other==_sources;
    }
private:

    LSST_PERSIST_FORMATTER(lsst::afw::formatters::SourceVectorFormatter)
    SourceSet _sources;
}; 


}}}  // namespace lsst::afw::detection

#ifndef SWIG
BOOST_CLASS_VERSION(lsst::afw::detection::Source, 2)
#endif

#endif // LSST_AFW_DETECTION_SOURCE_H

