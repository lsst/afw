// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_UTILS_H
#define LSST_AFW_MATH_UTILS_H
/**
 * \file
 *
 * \brief Math utilities.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <cmath>

#include <lsst/afw/math/Function.h>

namespace lsst {
namespace afw {
namespace math {
    
    /**
     * @brief Make a vector from a small collection of arguments
     */
    template<typename T>
    std::vector<T> makeVector(T elt0) {
        std::vector<T> vec;
        vec.push_back(elt0);
        return vec;
    }
    
    template<typename T>
    std::vector<T> makeVector(T elt0, T elt1) {
        std::vector<T> vec;
        vec.push_back(elt0);
        vec.push_back(elt1);
        return vec;
    }
    
    template<typename T>
    std::vector<T> makeVector(T elt0, T elt1, T elt2) {
        std::vector<T> vec;
        vec.push_back(elt0);
        vec.push_back(elt1);
        vec.push_back(elt2);
        return vec;
    }
    
    template<typename T>
    std::vector<T> makeVector(T elt0, T elt1, T elt2, T elt3) {
        std::vector<T> vec;
        vec.push_back(elt0);
        vec.push_back(elt1);
        vec.push_back(elt2);
        vec.push_back(elt3);
        return vec;
    }
    
    template<typename T>
    std::vector<T> makeVector(T elt0, T elt1, T elt2, T elt3, T elt4) {
        std::vector<T> vec;
        vec.push_back(elt0);
        vec.push_back(elt1);
        vec.push_back(elt2);
        vec.push_back(elt3);
        vec.push_back(elt4);
        return vec;
    }

}}}   // lsst::afw::math

#endif // #ifndef LSST_AFW_MATH_UTILS_H
