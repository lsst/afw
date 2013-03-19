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

/**
 * @file GaussianProcess.h
 *
 * @ingroup afw
 *
 * @author Scott Daniel
 * Contact scott.f.daniel@gmail.com
 *
 * Created in February 2013
*/

#include <Eigen/Dense>
#include "ndarray/eigen.h"

namespace lsst {
namespace afw {
namespace math {

/**
 *@class KdTree
 *
 * @brief The data for GaussianProcess is stored in a KD tree to facilitate nearest-neighbor searches
 *
 * @ingroup afw
 *
 * Important member variables are
 *   
 * _tree is a list of integers defining the structure of the tree.
 * _tree[i][0] is the dimension on which the ith point divides its daughters
 * _tree[i][1] is the index of the left hand daughter of the ith point (_data[_tree[i][1]][_tree[i][0]] < _data[i][_tree[i][0]])
 * _tree[i][2] is the index of the right hand daughter of the ith point (_data[_tree[i][2]][_tree[i][0]] >= _data[i][_tree[i][0]])
 * _tree[i][3] is the index of the parent of the ith point
 *
 *  _pts the number of data points stored in this tree
 *
 * _dimensions is the dimensionality of the parameter space on which the tree is defined
 *
 * _room is the number of points allotted in _tree and data (room will often be larger than _pts so that
 * new and delete do not have to be called every time a new point is added to the tree)
 *
 * data stores the data points in the tree.  data[i][j] is the jth element of the ith data point
*/

template <typename T>
class KdTree{


 public:
    ndarray::Array<T,2,2> data;
 
    ~KdTree();
    
    
  /**
    * @brief Build a KD Tree to store the data for GaussianProcess
    *
    * @param [in] dd the number of dimensions of parameter space
    *
    * @param [in] pp the number of data points being read in
    *
    * @param [in] dt an array, the rows of which are the data points (dt[i][j] is the jth component of the ith data point)
    *
    * @param [in] dfn a function defining the distance by which KdTree will define ``nearest neighbors''
   */
    KdTree(int dd,int pp,ndarray::Array<T,2,2> const &dt,\
    double(*dfn)(ndarray::Array<T,1,1> const &,ndarray::Array<T,1,1> const &,int));
    
    /**
     * @brief Find the nearest neighbors of a point
     *
     * @param [in] v the point whose neighbors you want to find
     *
     * @param [in] n_nn the number of nearest neighbors you want to find
     *
     * @param [in,out] neighdex this is where the indices of the nearest neighbor points will be stored
     *
     * @param [in,out] dd this is where the distances to the nearest neighbors will be stored
     *
     * neighbors will be returned in ascending order of distance
     *
     * note that distance is defined by the function which was passed into the constructor
  
    */
    void findNeighbors(ndarray::Array<T,1,1> const &v,int n_nn,\
    ndarray::Array<int,1,1> neighdex,ndarray::Array<double,1,1> dd);
    
    /**
     * @brief Add a point to the tree.  Allot more space in _tree and data if needed.
     *
     * @param [in] v the point you are adding to the tree
   */
    void addPoint(ndarray::Array<T,1,1> const &v);

    /**
     * @brief return the number of data points stored in the tree
    */
    int getPoints();
    
   /**
     * @brief Return the _tree information for a given data point
   */
    void getTreeNode(int,ndarray::Array<int,1,1>);
    
   /**
     * @brief Make sure that the tree is properly constructed.  Returns 1 of it is.
   */
    int testTree();
     
 private:
    ndarray::Array<int,2,2> _tree;
    ndarray::Array<int,1,1> _inn;
   
    int _pts,_dimensions,_room,_roomStep,_masterParent;
    int _neighborsFound,_neighborsWanted;
    ndarray::Array<T,1,1> _toSort;
    
    ndarray::Array<double,1,1> _neighborDistances;
    ndarray::Array<int,1,1> _neighborCandidates;
       
    /**
     * @brief Find the daughter point of a node in the tree and segregate the points around it
     * 
     * @param [in] use the indices of the data points being considered as possible daughters
     *
     * @param [in] ct the number of possible daughters
     *
     * @param [in] parent the index of the parent whose daughter we are chosing
     *
     * @param [in] dir which side of the parent are we on?  dir==1 means that we are on the left side; dir==2 means the right side.
    */
    void _organize(ndarray::Array<int,1,1> const &use,int ct,int parent,int dir);
       
    /**
      * @brief Find the point already in the tree that would be the parent of a point not in the tree
      *
      * @param [in] v the points whose prospective parent you want to find
    */
    int _findNode(ndarray::Array<T,1,1> const &v);
    
   /**
    * @brief This method actually looks for the neighbors, determining whether or not to descend branches of the tree
    *
    * @param [in] v the point whose neighbors you are looking for
    *
    * @param [in] consider the index of the data point you are considering as a possible nearest neighbor
    *
    * @param [in] from the index of the point you last considered as a nearest neighbor (so the search does not backtrack along the tree)
    *
    * The class KdTree keeps track of how many neighbors you want and how many neighbors you have found and what their
    * distances from v are in the class member variables _neighborsWanted, _neighborsFound, _neighborCandidates,
    * and _neighborDistances
   */
    void _lookForNeighbors(ndarray::Array<T,1,1> const &v,int consider,int from);
    
    /**
     * @brief A method to make sure that every data point in the tree is in the correct relation to its parents
    */
    int _walkUpTree(int,int,int);
    
    double (*_distance)(ndarray::Array<T,1,1> const &,\
    ndarray::Array<T,1,1> const &,int);
    
};


/**
 * @brief Stores values of a function sampled on an image and allows you to interpolate the function to unsampled points
 *
 * @ingroup afw
 *
 * The data will be stored in a KD Tree for easy nearest neighbor searching when interpolating.
 *
 * The array function[] will contain the values of the function being interpolated.
 *
 * _data[i][j] will be the jth component of the ith data point.  _function[i] is the value of the
 * function at the ith data point.
 *
 * _max and _min contain the maximum and minimum values of each dimension in parameter space
 * (if applicable) so that data points can be normalized by _max-_min to keep distances between
 * points reasonable.  This is an option specified by calling the relevant constructor.
 *
*/

template <typename T>
class GaussianProcess{

 public:
  
    double interpolationTime,neighborSearchTime,inversionTime;
    double iterationTime,varSolveTime;
    int interpolationCount;
     
    enum{squaredExp,neuralNetwork};
    
    ~GaussianProcess();
    
    /**
     @brief This is the constructor you call if you do not wish to normalize the positions of your data points
      *
      * @param [in] dd the number of dimensions of your data points
      *
      * @param [in] pp the number of data points you are inputting
      *
      * @param [in] datain an ndarray containing the data points; the ith row of datain is the ith data point
      *
      * @param [in] ff a one-dimensional ndarray containing the values of the scalar function associated with each data point.  
      * This is the function you are interpolating
    */
    GaussianProcess(int dd,int pp,ndarray::Array<T,2,2> const &datain,ndarray::Array<T,1,1> const &ff);
     
    /**
     * @brief This is the constructor you call if you want the positions of your data points normalized by the span of each dimension
     *
     * @param [in] dd the number of dimensions of your data points
     *
     * @param [in] pp the number of data points you are inputting
     *
     * @param [in] datain an ndarray containing the data points; the ith row of datain is the ith data point
     *
     * @param [in] mn a one-dimensional ndarray containing the minimum values of each dimension (for normalizing the positions of data points)
     *
     * @param [in] mx a one-dimensional ndarray containing the maximum values of each dimension (for normalizing the positions of data points)
     *
     * @param [in] ff a one-dimensional ndarray containing the values of the scalar function associated with each data point.  
     * This is the function you are interpolating
     *
     *Note: the member variable _useMaxMin will allow the code to remember which constructor you invoked
    */
    GaussianProcess(int dd,int pp,\
    ndarray::Array<T,2,2> const &datain,ndarray::Array<T,1,1> const &mn,ndarray::Array<T,1,1> const &mx,\
    ndarray::Array<T,1,1> const &ff);
     
    /**
     @brief Interpolate the function value at one point using a specified number of nearest neighbors
     * 
     * @param [in] vin a one-dimensional ndarray representing the point at which you want to interpolate the function
     *
     * @param [out] variance a one-dimensional ndarray.  The value of the variance predicted by the Gaussina process will be stored in the zeroth element
     *
     * @param [in] kk the number of nearest neighbors to be used in the interpolation
     *
     * the interpolated value of the function will be returned at the end of this method
     *
     *Note: the member variable _useMaxMin will allow the code to remember which constructor you invoked
    */
    T interpolate(ndarray::Array<T,1,1> const &vin,ndarray::Array<T,1,1> variance,int kk);
     
    /**
     * @brief This method will interpolate the function on a data point for purposes of optimizing hyper parameters
     *
     * @param [in] dex the index of the point you wish to self interpolate
     *
     * @param [out] variance a one-dimensional ndarray.  The value of the variance predicted by the Gaussina process will be stored in the zeroth element
     *
     * @param [in] kk the number of nearest neighbors to be used in the interpolation
     *
     * The interpolated value of the function will be returned at the end of this method
     *
     * This method ignores the point on which you are interpolating when requesting nearest neighbors
     *
    */
    T selfInterpolate(int dex,ndarray::Array<T,1,1> variance,int kk);

    /**
     * @brief Interpolate a list of query points using all of the input data (rather than nearest neighbors)
     *
     * @param [in] queries a 2-dimensional ndarray containing the points to be interpolated.  queries[i][j] is the jth component of the ith point
     *
     * @param [out] mu a 1-dimensional ndarray where the interpolated function values will be stored
     *
     * @param [out] variance a 2-dimensional ndarray where the corresponding variances in the function value will be stored
     *
     * @param [in] nQueries the number of points being interpolated
     *
     * This method will attempt to construct a _pts X _pts covariance matrix C and solve the problem Cx=b.
     * Be wary of using it in the case where _pts is very large.
     *
     * This version of the method will also return variances for all of the query points.  That is a very time consuming
     * calculation relative to just returning estimates for the function.  Consider calling the version of this method
     * that does not calculate variances (below).  The difference in speed is a factor of two in the case of
     * 189 data points and 1 million queries.
     *
    */
    void batchInterpolate(ndarray::Array<T,2,2> const &queries,ndarray::Array<T,1,1> mu,ndarray::Array<T,1,1> variance,\
    int nQueries);
     
    /**
     * @brief Interpolate a list of points using all of the data. Do not return variances for the interpolation.
     *
     * @param [in] queries a 2-dimensional ndarray containing the points to be interpolated.  queries[i][j] is the jth component of the ith point
     *
     * @param [out] mu a 1-dimensional ndarray where the interpolated function values will be stored
     *
     * @param [in] nQueries the number of points being interpolated
     *
     * This method will attempt to construct a _pts X _pts covariance matrix C and solve the problem Cx=b.
     * Be wary of using it in the case where _pts is very large.
     *
     * This version of the method does not return variances.  It is an order of magnitude faster than the version of the method
     * that does return variances (timing done on a case with 189 data points and 1 million query points).
     *
    */
    void batchInterpolate(ndarray::Array<T,2,2> const &queries,ndarray::Array<T,1,1> mu,int nQueries);

    /**
     * @brief Add a point to the pool of data used by GaussianProcess for interpolation
     *
     * @param [in] vin a one-dimensional ndarray storing the point in parameter space that you are adding
     *
     * @param [in]  f the value of the function at that point
    */
    void addPoint(ndarray::Array<T,1,1> const &v,T f);
     
    /**
     * @brief Assign a value to the Kriging paramter
     *
     * @param [in] kk the value assigned to the Kriging parameters
     *
    */
    void setKrigingParameter(T);
   
    /**
     * @brief set the value of the hyperparameter _lambda
     *
     * @param [in] ll the value you want assigned to _lambda
     *
     * _lambda is a parameter meant to represent the characteristic variance
     * of the function you are interpolating.  Currently, it is a scalar such that
     * all data points must have the same characteristic variance.  Future iterations
     * of the code may want to promote _lambda to an array so that different data points
     * can have different variances.
    */
    void setLambda(T ll); 
      
    /**
     * @brief Set the values of the hyperparameters governing the covariogram.  The method knows how many there should be.  
     *
     * @param [in] hyin a one-dimensional ndarray containing the hyperparameter values to be set.
     * 
     * The number of parameters in hyin should correspond to the number of parameters associated with the chosen type
     * of covariogram
    */
    void setHyperParameters(ndarray::Array<double,1,1> const &hyin);
    
    /**
     * @brief Select the type of covariogram from those enumerated in above
     *
     * @param [in] ii The type of covariogram you want to use
     *
     * At this point, supported types are
     *
     * GaussianProcess::squaredExp -- the squared exponent covariogram
     *
     * GaussianProcess::neuralNetwork -- the covariogram of a neural network with infinite hidden layers
     * see Rasmussen and Williams (2006), http://gaussianprocess.org/gpml/    equation 4.29 
     *
     * If you give it an unkown option, the code will just set the squared exponent covariogram
     *
     * This method automatically sets the size of _hyperParameters to whatever is appropriate
    */
    void setCovariogramType(int ii);
    
    /**
     * @brief Output the indices of data points curently stored in the _neighbors array
     *
     *@param [out] v an array to store the requested neighbors
    */
    void getNeighbors(ndarray::Array<int,1,1> v);
     
    /**
     * @brief Output a specified row of the last computed covariance matrix
     *
     * @param [in] dex the row that you want
     *
     * @param [out] v a one-dimensiona ndarray to store the row
    */
    void getCovarianceRow(int,ndarray::Array<T,1,1>);
     
    /**
     * @brief Run KdTree::testTree to make sure that the KD Tree is properly constructed.  Returns 1 if it is.
    */
    int testKdTree();
     
    /**
     * @brief Print the the time spent on neighbor searches, interpolation, matrix inversion, iterating over matrix indices, and finding variances
    */
    void getTimes();
     
    /**
     * @brief Reset the times being tracked inside interpolate
    */
    void resetTimes();
     
    void waste(ndarray::Array<T,2,2> const &);
     
 private:
    int _pts,_numberOfNeighbors,_useMaxMin,_dimensions,_room,_roomStep;
    int _calledInterpolate,_nHyperParameters,_typeOfCovariogram;
    
    ndarray::Array<int,1,1> _neighbors;
    ndarray::Array<double,1,1>_neighborDistances;
    
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> matrixtype;
    

    T _krigingParameter,_lambda;

    ndarray::Array<T,1,1> _function,_covarianceTestPoint;
    ndarray::Array<T,1,1> _vv,_max,_min;
    ndarray::Array<double,1,1> _hyperParameters;
    ndarray::Array<T,2,2> _data;
    
    Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> _covariance,_covarianceInverse;
    Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> _bb,_xx;
    
    Eigen::LLT<matrixtype> _llt;
    
    KdTree<T> *_kdTreePtr;
          
    double (*_distance)(ndarray::Array<T,1,1> const &,ndarray::Array<T,1,1> const &,int);
    
    T (*_covariogram)(ndarray::Array<T,1,1> const &,ndarray::Array<T,1,1> const & \
    ,int,ndarray::Array<double,1,1> const &);

};

}}}
