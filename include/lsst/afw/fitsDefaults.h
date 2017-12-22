// -*- lsst-c++ -*-
#ifndef LSST_AFW_fitsDefaults_h_INCLUDED
#define LSST_AFW_fitsDefaults_h_INCLUDED

#include <climits>

namespace lsst {
namespace afw {
namespace fits {

/**
 *  Specify that the default HDU should be read.
 *
 *  This special HDU number indicates that the first extension
 *  should be used if the primary HDU is empty (i.e., has NAXIS=0)
 *  and the Primary HDU is the current.
 */
const int DEFAULT_HDU = INT_MIN;

}}} // namespace lsst::afw::fits

#endif