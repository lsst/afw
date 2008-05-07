// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Define methods for Function classes.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <sstream>

#include "lsst/afw/math.h"

template<typename ReturnT>
std::string lsst::afw::math::Function<ReturnT>::toString(void) const {
    std::stringstream os;
    os << this->_name << "(";
    bool isFirst = true;
    for (std::vector<double>::const_iterator i = _params.begin(); i != _params.end(); ++i) {
        if (!isFirst) {
            os << ", ";
        } else {
            isFirst = false;
        }
        os << *i;
    }
    os << ")";
    return os.str();
};

template<typename ReturnT>
std::string lsst::afw::math::SeparableFunction<ReturnT>::toString(void) const {
    std::stringstream os;
    os << this->_name << "(";
    bool isFirst = true;
    for (typename functionListType::const_iterator funcIter = _functionList.begin();
         funcIter != _functionList.end();  ++funcIter) {
        if (!isFirst) {
            os << ", ";
        } else {
            isFirst = false;
        }
        os << (*funcIter)->toString();
    }
    os << ")";
    return os.str();
};

/************************************************************************************************************/
//
// Explicit instantiations
//
template class lsst::afw::math::Function<float>;
template class lsst::afw::math::Function1<float>;
template class lsst::afw::math::Function2<float>;
template class lsst::afw::math::SeparableFunction<float>;
template class lsst::afw::math::SeparableFunction2<float>;

template class lsst::afw::math::Function<double>;
template class lsst::afw::math::Function1<double>;
template class lsst::afw::math::Function2<double>;
template class lsst::afw::math::SeparableFunction<double>;
template class lsst::afw::math::SeparableFunction2<double>;
