#ifndef LSST_AFW_MATH_COORDINATE_H
#define LSST_AFW_MATH_COORDINATE_H

#include <Eigen/Core>

namespace lsst {
namespace afw {
namespace math {

typedef Eigen::Matrix<double,2,1,Eigen::RowMajor | Eigen::DontAlign> Coordinate;

}}} //namespace lsst::afw::math;

#endif 
