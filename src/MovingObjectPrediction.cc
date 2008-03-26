// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file   MovingObjectPrediction.cc
//!
//##====----------------                                ----------------====##/

#include "lsst/fw/MovingObjectPrediction.h"

namespace lsst {
namespace fw {


// -- MovingObjectPrediction ----------------

MovingObjectPrediction::MovingObjectPrediction() :
    _orbitId( -1),
    _ra     (0.0),
    _dec    (0.0),
    _smaa   (0.0),
    _smia   (0.0),
    _pa     (0.0),
    _mjd    (0.0),
    _mag    (0.0),
    _magErr (0.0)
{}


bool MovingObjectPrediction::operator==(MovingObjectPrediction const & d) const {
    if (this == &d) {
        return true;
    }
    return _orbitId == d._orbitId &&
           _ra      == d._ra      &&
           _dec     == d._dec     &&
           _smaa    == d._smaa    &&
           _smia    == d._smia    &&
           _pa      == d._pa      &&
           _mjd     == d._mjd     &&
           _mag     == d._mag     &&
           _magErr  == d._magErr;
}


// -- MovingObjectPredictionVector ----------------

MovingObjectPredictionVector::MovingObjectPredictionVector() :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec()
{}


MovingObjectPredictionVector::MovingObjectPredictionVector(size_type n) :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec(n)
{}


MovingObjectPredictionVector::MovingObjectPredictionVector(size_type n, value_type const & val) :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec(n, val)
{}


MovingObjectPredictionVector::~MovingObjectPredictionVector() {}


MovingObjectPredictionVector::MovingObjectPredictionVector(MovingObjectPredictionVector const & v) :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec(v._vec)
{}


MovingObjectPredictionVector::MovingObjectPredictionVector(Vector const & v) :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec(v)
{}


MovingObjectPredictionVector & MovingObjectPredictionVector::operator=(
    MovingObjectPredictionVector const & v
) {
    if (this != &v) {
        _vec = v._vec;
    }
    return *this;
}

MovingObjectPredictionVector & MovingObjectPredictionVector::operator=(Vector const & v) {
    _vec = v;
    return *this;
}


}} // end of namespace lsst::fw

