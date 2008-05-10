// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Define methods for Function classes.
 *
 * \todo
 * The source code has not been updated to reflect the new Function.h design.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <sstream>

#include "boost/format.hpp"

#include "lsst/afw/math.h"

/**
 * \brief Format function parameters as a comma-separated list
 *
 * \return parameters formatted as: "name0=val, name1=val1, ..."
 */
template<typename ReturnT>
std::string formatParameters(lsst::afw::math::Function<ReturnT> const & func) {
    std::stringstream os;
    unsigned int nParams = func.getNParameters();
    std::vector<std::string> paramNames = func.getParameterNames();
    if (paramNames.size() < nParams) {
        for (unsigned int i = paramNames.size(); i < nParams; ++i) {
            paramNames.push_back(boost::str(boost::format("P%u") % i));
        }
    }
    bool isFirst = true;
    for (unsigned int i = 0; i < nParams; ++i) {
        if (!isFirst) {
            os << ", ";
        } else {
            isFirst = false;
        }
        os << paramNames[i];
        os << "=";
        os << func.getParameter(i);
    }
    return os.str();
}

template<typename ReturnT>
std::vector<std::string> lsst::afw::math::Function<ReturnT>::getParameterNames(void) const {
    unsigned int nParams = this->getNParameters();
    std::vector<std::string> paramNames;
    for (unsigned int i = 0; i < nParams; ++i) {
        paramNames.push_back(boost::str(boost::format("P%u") % i));
    }
    return paramNames;
}

template<typename ReturnT>
std::string lsst::afw::math::Function<ReturnT>::toString(void) const {
    std::stringstream os;
    os << this->_name << "()[" << formatParameters(*this) << "]";
    return os.str();
};


/************************************************************************************************************/
//
// Explicit instantiations
//
template class lsst::afw::math::Function<float>;
template class lsst::afw::math::Function1<float>;
template class lsst::afw::math::Function2<float>;
template class lsst::afw::math::SeparableFunction2<float>;

template class lsst::afw::math::Function<double>;
template class lsst::afw::math::Function1<double>;
template class lsst::afw::math::Function2<double>;
template class lsst::afw::math::SeparableFunction2<double>;
