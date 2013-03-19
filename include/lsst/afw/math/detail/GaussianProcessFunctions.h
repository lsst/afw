// -*- LSST-C++ -*-

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

#include "ndarray/eigen.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace GaussianProcess {

/** 
 * This namespace contains some functions that need to be `global'
 * so that both GaussianProcess and KdTree can access them
*/

/**
 * @brief Return the Euclidean distance between two points in arbitrary-dimensional space
 *
 * This is left as a method in case future developers want to enable the possibility that
 * KdTree will define nearest neighbors by some other distance
 *
 * @param [in] v1 the first point
 *
 * @param [in] v2 the second point
 *
 * @param [in] d_dim the number of dimensions in the parameter space
 *
*/
template <typename T>
double euclideanDistance(ndarray::Array<T,1,1> const &v1,ndarray::Array<T,1,1> const &v2,int d_dim);

/**
 * @brief The squared exponential covariogram for GaussianProcess
 *
 * Takes two points in parameter space and returns the covariogram relation
 * between them
 *
 * @param [in] v1 the first point
 *
 * @param [in] v2 the second point
 *
 * @param [in] d_dim the number of dimensions in parameter space
 *
 * @param [in] hyp a list of hyperparameters governing the shape of the covariogram
 *
 * in this case, there is only one hyperparameter: the characteristic length scale squared
 *
*/
template <typename T>
T expCovariogram(ndarray::Array<T,1,1> const &v1,ndarray::Array<T,1,1> const &v2,\
int d_dim,ndarray::Array<double,1,1> const &hyp);

/**
 * @brief The covariogram of a neural network with infinite hidden layers
 *
 * See Chapter 4 of Rasmussen and Williams (2006)
 * http://gaussianprocess.org/gpml/
 * equation (4.29)
 *
 * @param [in] v1 the first point
 *
 * @param [in] v2 the second point
 *
 * @param [in] d_dim the number of dimensions in parameter space
 *
 * @param [in] hyp a list of hyperparameters governing the shape of the covariogram
 *
 * in this case, there are two hyper parameters as defined by Rasmussen and Williams
 * (they call them \sigma^2 and \sigma^2_0)
 *
*/
template <typename T>
T neuralNetCovariogram(ndarray::Array<T,1,1> const &v1,ndarray::Array<T,1,1> const &v2,int d_dim,\
ndarray::Array<double,1,1> const &hyp);

/**
 * @brief Sort a list of numbers using a merge sort algorithm
 *
 * mergeSort is the `outer' method which implements the merge sort
 * algorithm from Numerical Recipes.  It relies on mergeScanner
 * to be complete
 *
 * @param [in] insort is the list of numbers to be sorted
 *
 * @param [in] indices Keeps track of their original order (in case there is another
 * list that needs to be correlated with the list of sorted values)
 *
 * @param [in] el is the number of values being sorted
 *
*/
template<typename T>
void mergeSort(ndarray::Array<T,1,1> const &insort,ndarray::Array<int,1,1> const &indices,int el);


/**
 *@brief Rearrange the elements of a list to facilitate merge sorting
 *
 * mergeScanner will take the matrix m and put everything in it with value
 * greater than element m[dex] to the right of that element
 * and everything less than m[dex] to the left; it then returns
 * the new index of that anchored element (which you now *know* is in
 * the right spot
 
 * It is part of an implemenation of the merge sort algorithm described
 * in Numerical Recipes (2nd edition); Press, Teukolsky, Vetterling, and Flannery
 * 1992
  
 * @param [in] m is a list of numbers to be sorted
 *
 * @param [in] indices is a list of ints which keeps track of the sorted numbers original
 * positions
 * 
 * @param [in] dex denotes the value about which everything is to be sorted
 * (i.e. values less than m[dex] will get put to the left of it;
 * values greater than m[dex] will get put to the right)
 *
 * @param [in] el denotes how many elements are in m[] and indices[]
*/
template<typename T>
int mergeScanner(ndarray::Array<T,1,1> const &m,ndarray::Array<int,1,1> const &indices,\
int dex,int el);

}}}}}

