// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */


#ifndef LSST_AFW_MATH_GAUSSIAN_PROCESS_H
#define LSST_AFW_MATH_GAUSSIAN_PROCESS_H

#include <Eigen/Dense>
#include <stdexcept>

#include "ndarray/eigen.h"
#include "boost/shared_ptr.hpp"

#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/DateTime.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace math {

/**
  * @class GaussianProcessTimer
  *
  * @brief This is a structure for keeping track of how long the 
  * interpolation methods spend on different parts of the interpolation
  *
  * _eigenTime keeps track of how much time is spent using Eigen's linear algebra packages
  *
  * _iterationTime keeps track of how much time is spent iterating over matrix indices
  * (this is also a catch-all for time that does not obviously fit in the other categories)
  *
  * _searchTime keeps track of how much time is spent on nearest neighbor searches (when applicable)
  *
  * _varianceTime keeps track of how much time is spent calculating the variance of our 
  * interpolated function value (note: time spent using Eigen packages for this purpose
  * is tallied here, not in _eigenTime)
  *
  * _totalTime keeps track of how much time total is spent on interpolations
  *
  * _interpolationCount keeps track of how many points have been interpolated
*/
class GaussianProcessTimer{

public:

    GaussianProcessTimer();

    /**
     * @brief Resets all of the data members of GaussianProcessTimer to zero.
     *
    */
    void reset();

    /**
     * @brief Starts the timer for an individual call to an interpolation routine
    */
    void start();

    /**
     * @brief Adds time to _eigenTime
    */
    void addToEigen();

    /**
     * @brief Adds time to _varianceTime
    */
    void addToVariance();

    /**
     * @brief Adds time to _searchTime
    */
    void addToSearch();

    /**
     * @brief Adds time to _iterationTime
    */
    void addToIteration();

    /**
     * @brief Adds time to _totalTime and adds counts to _interpolationCount
     *
     * @param [in] i the number of counts to add to _interpolationCount
    */
    void addToTotal(int i);

    /**
     * @brief Displays the current values of all times and _interpolationCount
    */
    void display();

private:
    double _before,_beginning;
    double _eigenTime,_iterationTime,_searchTime,_varianceTime,_totalTime;
    int _interpolationCount;
};


/**
  * @class Covariogram
  *
  * @brief The parent class of covariogram functions for use in Gaussian Process interpolation
  *
  * Each instantiation of a Covariogram will store its own hyper parameters
  *
*/
template <typename T>
class Covariogram : public lsst::daf::base::Citizen
                   #ifndef SWIG
                    , private boost::noncopyable
                    #endif
{
public:
    virtual ~Covariogram();
   
    /**
     * @brief construct a Covariogram assigning default values to the hyper parameters
    */
    explicit Covariogram():lsst::daf::base::Citizen(typeid(this)){};


    /**
     * @brief Actually evaluate the covariogram function relating two points you want to interpolate from
     *
     * @param [in] p1 the first point
     *
     * @param [in] p2 the second point
    */
    virtual T operator() (ndarray::Array<const T,1,1> const &p1,
                          ndarray::Array<const T,1,1> const &p2
                          ) const;

};

/**
 * @class SquaredExpCovariogram
 *
 * @brief a Covariogram that falls off as the negative exponent of the square 
 * of the distance between the points
 *
 * Contains one hyper parameter (_ellSquared) encoding the square of the
 * characteristic length scale of the covariogram
*/
template <typename T>
class SquaredExpCovariogram : public Covariogram<T>{

public:
    virtual ~SquaredExpCovariogram();

    explicit SquaredExpCovariogram();

    /**
     * @brief set the _ellSquared hyper parameter (the square of the characteristic 
     * length scale of the covariogram)
    */
    void setEllSquared(double ellSquared);

    virtual T operator() (ndarray::Array<const T,1,1> const &,
                          ndarray::Array<const T,1,1> const &
                          ) const;

private:
    double _ellSquared;

};

/**
  * @class NeuralNetCovariogram
  *
  * @brief a Covariogram that recreates a neural network with one hidden layer 
  * and infinite units in that layer
  *
  * Contains two hyper parameters (_sigma0 and _sigma1) that characterize the expected
  * variance of the function being interpolated
  *
  * see Rasmussen and Williams (2006) http://www.gaussianprocess.org/gpml/
  * equation 4.29
*/
template <typename T>
class NeuralNetCovariogram : public Covariogram<T>{

public:
    virtual ~NeuralNetCovariogram();

    explicit NeuralNetCovariogram();

    /**
     * @brief set the _sigma0 hyper parameter
    */
    void setSigma0(double sigma0);

    /**
     * @brief set the _sigma1 hyper parameter
    */
    void setSigma1(double sigma1);

    virtual T operator() (ndarray::Array<const T,1,1> const &,
                          ndarray::Array<const T,1,1> const &
                          ) const;

private:
    double _sigma0,_sigma1;


};

/**
 *@class KdTree
 *
 * @brief The data for GaussianProcess is stored in a KD tree to facilitate nearest-neighbor searches
 *
 * Note: I have removed the ability to arbitrarily specify a distance function.  The KD Tree nearest neighbor
 * search algorithm only makes sense in the case of Euclidean distances, so I have forced KdTree to use
 * Euclidean distances.
*/

template <typename T>
class KdTree
        #ifndef SWIG
        : private boost::noncopyable
        #endif
{
public:

    /**
     * @brief Build a KD Tree to store the data for GaussianProcess
     *
     * @param [in] dt an array, the rows of which are the data points 
     * (dt[i][j] is the jth component of the ith data point)
     *
     * @throw pex_exceptions RuntimeError if the tree is not properly constructed
    */
    void Initialize(ndarray::Array<T,2,2> const &dt);

    /**
     * @brief Find the nearest neighbors of a point
     *
     * @param [out] neighdex this is where the indices of the nearest neighbor points will be stored
     *
     * @param [out] dd this is where the distances to the nearest neighbors will be stored
     *
     * @param [in] v the point whose neighbors you want to find
     *
     * @param [in] n_nn the number of nearest neighbors you want to find
     *
     * neighbors will be returned in ascending order of distance
     *
     * note that distance is forced to be the Euclidean distance
    */
    void findNeighbors(ndarray::Array<int,1,1> neighdex,
                       ndarray::Array<double,1,1> dd,
                       ndarray::Array<const T,1,1> const &v,
                       int n_nn) const;


    /**
     * @brief Return one element of one node on the tree
     *
     * @param [in] ipt the index of the node to return
     *
     * @param [in] idim the index of the dimension to return
    */
    T getData(int ipt, int idim) const;


    /**
     * @brief Return an entire node from the tree
     *
     * @param [in] ipt the index of the node to return
     *
     * I currently have this as a return-by-value method.  When I tried it as
     * a return-by-reference, the compiler gave me
     *
     * warning: returning reference to local temporary object
     *
     * Based on my reading of Stack Overflow, this is because ndarray
     * was implicitly creating a new ndarray::Array<T,1,1> object and passing
     * a reference thereto.  It is unclear to me whether or not this object
     * would be destroyed once the call to getData was complete.
     *
     * The code still compiled, ran, and passed the unit tests, but the above
     * behavior seemed to me like it could be dangerous (and, because ndarray
     * was still creating a new object, it did not seem like we were saving
     * any time), so I reverted to return-by-value.
     *
    */
    ndarray::Array<T,1,1> getData(int ipt) const;

    /**
     * @brief Add a point to the tree.  Allot more space in _tree and data if needed.
     *
     * @param [in] v the point you are adding to the tree
     *
     * @throw pex_exceptions RuntimeError if the branch ending in the new point is not properly constructed
    */
    void addPoint(ndarray::Array<const T,1,1> const &v);

    /**
     * @brief Remove a point from the tree.  Reorganize what remains so that the tree remains self-consistent
     *
     * @param [in] dex the index of the point you want to remove from the tree
     *
     * @throw pex_exceptions RuntimeError if the entire tree is not poperly constructed after 
     * the point has been removed
    */
    void removePoint(int dex);

    /**
     * @brief return the number of data points stored in the tree
    */
    int getPoints() const;

   /**
     * @brief Return the _tree information for a given data point
     *
     * @param [out] v the array in which to store the entry from _tree
     *
     * @param [in] dex the index of the node whose information you are requesting
   */
    void getTreeNode(ndarray::Array<int,1,1> const &v,int dex) const;


private:
    ndarray::Array<int,2,2> _tree;
    ndarray::Array<int,1,1> _inn;
    ndarray::Array<T,2,2> _data;

    enum{DIMENSION,LT,GEQ,PARENT};

    //_tree stores the relationships between data points
    //_tree[i][DIMENSION] is the dimension on which the ith node segregates its daughters
    //
    //_tree[i][LT] is the branch of the tree down which the daughters' DIMENSIONth component 
    //is less than the parent's
    //
    //_tree[i][GEQ] is the branch of the tree down which the daughters' DIMENSIONth component is 
    //greather than or equal to the parent's
    //
    //_tree[i][PARENT] is the parent node of the ith node

    //_data actually stores the data points


    int _pts,_dimensions,_room,_roomStep,_masterParent;
    mutable int _neighborsFound,_neighborsWanted;

    //_room denotes the capacity of _data and _tree.  It will usually be larger
    //than _pts so that we do not have to reallocate
    //_tree and _data every time we add a new point to the tree

    mutable ndarray::Array<double,1,1> _neighborDistances;
    mutable ndarray::Array<int,1,1> _neighborCandidates;

    /**
     * @brief Find the daughter point of a node in the tree and segregate the points around it
     *
     * @param [in] use the indices of the data points being considered as possible daughters
     *
     * @param [in] ct the number of possible daughters
     *
     * @param [in] parent the index of the parent whose daughter we are chosing
     *
     * @param [in] dir which side of the parent are we on?  dir==1 means that we are on the left
     * side; dir==2 means the right side.
    */
    void _organize(ndarray::Array<int,1,1> const &use,
                   int ct,
                   int parent,
                   int dir);

    /**
      * @brief Find the point already in the tree that would be the parent of a point not in the tree
      *
      * @param [in] v the points whose prospective parent you want to find
    */
    int _findNode(ndarray::Array<const T,1,1> const &v) const;

   /**
    * @brief This method actually looks for the neighbors, determining whether or 
    * not to descend branches of the tree
    *
    * @param [in] v the point whose neighbors you are looking for
    *
    * @param [in] consider the index of the data point you are considering as a possible nearest neighbor
    *
    * @param [in] from the index of the point you last considered as a nearest neighbor
    *  (so the search does not backtrack along the tree)
    *
    * The class KdTree keeps track of how many neighbors you want and how many 
    * neighbors you have found and what their distances from v are in the class member 
    * variables _neighborsWanted, _neighborsFound, _neighborCandidates,
    * and _neighborDistances
   */
    void _lookForNeighbors(ndarray::Array<const T,1,1> const &v,
                           int consider,
                           int from) const;

    /**
     * @brief Make sure that the tree is properly constructed.  Returns 1 of it is.  Return zero if not.
    */
    int _testTree() const;

    /**
     * @brief A method to make sure that every data point in the tree is in the
     * correct position relative to its parents
     *
     * @param [in] target is the index of the node you are looking at
     *
     * @param [in] dir is the direction (1,2) of the branch you ascended from root
     *
     * @param [in] root is the node you started walking up from
     *
     * This method returns the value of _masterParent if the branch is correctly contructed.
     * It returns zero otherwise
    */
    int _walkUpTree(int target,
                   int dir,
                   int root) const;

    /**
      * @brief A method which counts the number of nodes descended from a given node
      * (used by removePoint(int))
      *
      * @param [in] where the node you are currently on
      *
      * @param [in,out] *ct keeps track of how many nodes you have encountered as you descend the tree
    */
    void _count(int where,int *ct) const;

    /**
     * @brief Descend the tree from a node which has been removed, reassigning severed nodes as you go
     *
     * @param root the index of the node where you are currently
    */
    void _descend(int root);

    /**
      * @brief Reassign nodes to the tree that were severed by a call to removePoint(int)
      *
      * @param target the node you are reassigning
    */
    void _reassign(int target);

    /**
     * @brief calculate the Euclidean distance between the points p1 and p2
    */
    double _distance(ndarray::Array<const T,1,1> const &p1,
                     ndarray::Array<const T,1,1> const &p2) const;
};


/**
 * @class GaussianProcess
 *
 * @brief Stores values of a function sampled on an image and allows
 * you to interpolate the function to unsampled points
 *
 * The data will be stored in a KD Tree for easy nearest neighbor searching when interpolating.
 *
 * The array _function[] will contain the values of the function being interpolated.
 * You can provide a two dimensional array _function[][] if you wish to interpolate a vector of functions.  
 * In this case _function[i][j] is the jth function associated with the ith data point.  Note: presently, 
 * the covariance matrices do not relate elements of _function[i][]
 * to each other, so the variances returned will be identical for all functions evaluated at 
 * the same point in parameter space.
 *
 * _data[i][j] will be the jth component of the ith data point.
 *
 * _max and _min contain the maximum and minimum values of each dimension in parameter space
 * (if applicable) so that data points can be normalized by _max-_min to keep distances between
 * points reasonable.  This is an option specified by calling the relevant constructor.
 *
*/

template <typename T>
class GaussianProcess
                  #ifndef SWIG
                  : private boost::noncopyable
                  #endif
{

public:

    /**
      * @brief This is the constructor you call if you do not wish to normalize the positions
      * of your data points and you have only one function
      *
      * @param [in] dataIn an ndarray containing the data points; the ith row of datain is the ith data point
      *
      * @param [in] ff a one-dimensional ndarray containing the values 
      * of the scalar function associated with each data point. This is the 
      * function you are interpolating
      *
      * @param [in] covarIn is the input covariogram
    */
    GaussianProcess(ndarray::Array<T,2,2> const &dataIn,
                    ndarray::Array<T,1,1> const &ff,
                    boost::shared_ptr< Covariogram<T> > const &covarIn);

    /**
     * @brief This is the constructor you call if you want the positions of your data 
     * points normalized by the span of each dimension and you have only one function
     *
     * @param [in] dataIn an ndarray containing the data points; the ith row of datain is the ith data point
     *
     * @param [in] mn a one-dimensional ndarray containing the minimum values of each dimension 
     * (for normalizing the positions of data points)
     *
     * @param [in] mx a one-dimensional ndarray containing the maximum values of each dimension 
     * (for normalizing the positions of data points)
     *
     * @param [in] ff a one-dimensional ndarray containing the values of the scalar
     * function associated with each data point.  This is the function you are interpolating
     *
     * @param [in] covarIn is the input covariogram
     *
     * Note: the member variable _useMaxMin will allow the code to remember which constructor you invoked
    */
    GaussianProcess(ndarray::Array<T,2,2> const &dataIn,
                    ndarray::Array<T,1,1> const &mn,
                    ndarray::Array<T,1,1> const &mx,
                    ndarray::Array<T,1,1> const &ff,
                    boost::shared_ptr< Covariogram<T> > const &covarIn);
    /**
     * @brief this is the constructor to use in the case of a vector of input functions
     * and an unbounded/unnormalized parameter space
     *
     * @param [in] dataIn contains the data points, as in other constructors
     *
     * @param [in] ff contains the functions.  Each row of ff corresponds to a datapoint.
     *  Each column corresponds to a function (ff[i][j] is the jth function associated with
     *  the ith data point)
     *
     * @param [in] covarIn is the input covariogram
    */
    GaussianProcess(ndarray::Array<T,2,2> const &dataIn,
                    ndarray::Array<T,2,2> const &ff,
                    boost::shared_ptr< Covariogram<T> > const &covarIn);
    /**
     * @brief this is the constructor to use in the case of a vector of input
     * functions using minima and maxima in parameter space
     *
     * @param [in] dataIn contains the data points, as in other constructors
     *
     * @param [in] mn contains the minimum allowed values of the parameters in parameter space
     *
     * @param [in] mx contains the maximum allowed values of the parameters in parameter space
     *
     * @param [in] ff contains the functions.  Each row of ff corresponds to a datapoint.
     *  Each column corresponds to a function (ff[i][j] is the jth function associated with
     *  the ith data point)
     *
     * @param [in] covarIn is the input covariogram
    */
    GaussianProcess(ndarray::Array<T,2,2> const &dataIn,
                    ndarray::Array<T,1,1> const &mn,
                    ndarray::Array<T,1,1> const &mx,
                    ndarray::Array<T,2,2> const &ff,
                    boost::shared_ptr< Covariogram<T> > const &covarIn);

    /**
     * @brief Interpolate the function value at one point using a specified number of nearest neighbors
     *
     * @param [out] variance a one-dimensional ndarray.  The value of the variance predicted by the Gaussian
     * process will be stored in the zeroth element
     *
     * @param [in] vin a one-dimensional ndarray representing the point at which
     * you want to interpolate the function
     *
     * @param [in] numberOfNeighbors the number of nearest neighbors to be used in the interpolation
     *
     * the interpolated value of the function will be returned at the end of this method
     *
     * Note: if you used a normalized parameter space, you should not normalize 
     * vin before inputting.  The code will remember that you want a normalized 
     * parameter space, and will apply the normalization when you call interpolate
    */
    T interpolate(ndarray::Array<T,1,1> variance,
                  ndarray::Array<T,1,1> const &vin,
                  int numberOfNeighbors) const;
    /**
     * @brief This is the version of GaussianProcess::interpolate for a vector of functions.
     *
     * @param [out] mu will store the vector of interpolated function values
     *
     * @param [out] variance will store the vector of interpolated variances on mu
     *
     * @param [in] vin the point at which you wish to interpolate the functions
     *
     * @param [in] numberOfNeighbors is the number of nearest neighbor points to use in the interpolation
     *
     * Note: Because the variance currently only depends on the covariance function and the covariance
     * function currently does not include any terms relating different elements of mu to each other,
     * all of the elements of variance will be identical
    */
    void interpolate(ndarray::Array<T,1,1> mu,
                     ndarray::Array<T,1,1> variance,
                     ndarray::Array<T,1,1> const &vin,
                     int numberOfNeighbors) const;

    /**
     * @brief This method will interpolate the function on a data point 
     * for purposes of optimizing hyper parameters
     *
     * @param [out] variance a one-dimensional ndarray.  The value of the variance predicted by the
     * Gaussian process will be stored in the zeroth element
     *
     * @param [in] dex the index of the point you wish to self interpolate
     *
     * @param [in] numberOfNeighbors the number of nearest neighbors to be used in the interpolation
     *
     * @throw pex_exceptions RuntimeError if the nearest neighbor search does 
     * not find the data point itself as the nearest neighbor
     *
     * The interpolated value of the function will be returned at the end of this method
     *
     * This method ignores the point on which you are interpolating when requesting nearest neighbors
     *
    */
    T selfInterpolate(ndarray::Array<T,1,1> variance,
                      int dex,
                      int numberOfNeighbors) const;

    /**
     * @brief The version of selfInterpolate called for a vector of functions
     *
     * @param [out] mu this is where the interpolated function values will be stored
     *
     * @param [out] variance the variance on mu will be stored here
     *
     * @param [in] dex the index of the point you wish to interpolate
     *
     * @param [in] numberOfNeighbors the number of nearest neighbors to use in the interpolation
     *
     * @throw pex_exceptions RuntimeError if the nearest neighbor search does not find
     * the data point itself as the nearest neighbor
    */
    void selfInterpolate(ndarray::Array<T,1,1> mu,
                         ndarray::Array<T,1,1> variance,
                         int dex,
                         int numberOfNeighbors) const;

    /**
     * @brief Interpolate a list of query points using all of the input data (rather than nearest neighbors)
     *
     * @param [out] mu a 1-dimensional ndarray where the interpolated function values will be stored
     *
     * @param [out] variance a 1-dimensional ndarray where the corresponding variances
     * in the function value will be stored
     *
     * @param [in] queries a 2-dimensional ndarray containing the points to be interpolated.
     * queries[i][j] is the jth component of the ith point
     *
     * This method will attempt to construct a _pts X _pts covariance matrix C and solve the problem Cx=b.
     * Be wary of using it in the case where _pts is very large.
     *
     * This version of the method will also return variances for all of the query points.
     * That is a very time consuming calculation relative to just returning estimates for
     * the function.  Consider calling the version of this method that does not calculate
     * variances (below).  The difference in time spent is an order of magnitude for 189
     * data points and 1,000,000 interpolations.
     *
    */
    void batchInterpolate(ndarray::Array<T,1,1> mu,
                          ndarray::Array<T,1,1> variance,
                          ndarray::Array<T,2,2> const &queries) const;

    /**
     * @brief Interpolate a list of points using all of the data. Do not return variances for the
     * interpolation.
     *
     * @param [out] mu a 1-dimensional ndarray where the interpolated function values will be stored
     *
     * @param [in] queries a 2-dimensional ndarray containing the points to be interpolated.
     * queries[i][j] is the jth component of the ith point
     *
     * This method will attempt to construct a _pts X _pts covariance matrix C and solve the problem Cx=b.
     * Be wary of using it in the case where _pts is very large.
     *
     * This version of the method does not return variances.
     * It is an order of magnitude faster than the version of the method
     * that does return variances (timing done on a case with 189 data points and 1 million query points).
     *
    */
    void batchInterpolate(ndarray::Array<T,1,1> mu,
                          ndarray::Array<T,2,2> const &queries) const;

    /**
     * @brief This is the version of batchInterpolate (with variances) 
     * that is called for a vector of functions
    */
    void batchInterpolate(ndarray::Array<T,2,2> mu,
                          ndarray::Array<T,2,2> variance,
                          ndarray::Array<T,2,2> const &queries) const;

    /**
     * @brief This is the version of batchInterpolate (without variances) that
     * is called for a vector of functions
    */
    void batchInterpolate(ndarray::Array<T,2,2> mu,
                          ndarray::Array<T,2,2> const &queries) const;


    /**
     * @brief Add a point to the pool of data used by GaussianProcess for interpolation
     *
     * @param [in] vin a one-dimensional ndarray storing the point in parameter space that you are adding
     *
     * @param [in]  f the value of the function at that point
     *
     * @throw pex_exceptions RuntimeError if you call this when you should have
     * called the version taking a vector of functions (below)
     *
     * @throw pex_exceptions RuntimeError if the tree does not end up properly constructed
     * (the exception is actually thrown by KdTree<T>::addPoint() )
     *
     * Note: excessive use of addPoint and removePoint can result in an unbalanced KdTree,
     * which will slow down nearest neighbor searches
    */
    void addPoint(ndarray::Array<T,1,1> const &vin,T f);

    /**
     * @brief This is the version of addPoint that is called for a vector of functions
     *
     * @throw pex_exceptions RuntimeError if the tree does not end up properly constructed
     * (the exception is actually thrown by KdTree<T>::addPoint() )
     *
     * Note: excessive use of addPoint and removePoint can result in an unbalanced KdTree,
     * which will slow down nearest neighbor searches
    */
    void addPoint(ndarray::Array<T,1,1> const &vin,ndarray::Array<T,1,1> const &f);

    /**
     * @brief This will remove a point from the data set
     *
     * @param [in] dex the index of the point you want to remove from your data set
     *
     * @throw pex_exceptions RuntimeError if the tree does not end up properly constructed
     * (the exception is actually thrown by KdTree<T>::removePoint() )
     *
     * Note: excessive use of addPoint and removePoint can result in an unbalanced KdTree,
     * which will slow down nearest neighbor searches
    */
    void removePoint(int dex);

    /**
     * @brief Assign a value to the Kriging paramter
     *
     * @param [in] kk the value assigned to the Kriging parameters
     *
    */
    void setKrigingParameter(T kk);

    /**
     * @brief Assign a different covariogram to this GaussianProcess
     *
     * @param [in] covar the Covariogram object that you wish to assign
     *
    */
    void setCovariogram(boost::shared_ptr< Covariogram<T> > const &covar);

    /**
     * @brief set the value of the hyperparameter _lambda
     *
     * @param [in] lambda the value you want assigned to _lambda
     *
     * _lambda is a parameter meant to represent the characteristic variance
     * of the function you are interpolating.  Currently, it is a scalar such that
     * all data points must have the same characteristic variance.  Future iterations
     * of the code may want to promote _lambda to an array so that different data points
     * can have different variances.
    */
    void setLambda(T lambda);


    /**
     * @brief Give the user acces to _timer, an object keeping track of the time spent on
     * various processes within interpolate
     *
     * This will return a GaussianProcessTimer object.  The user can, for example, 
     * see how much time has been spent on Eigen's linear algebra package (see the
     * comments on the GaussianProcessTimer class) using code like
     *
     * gg=GaussianProcess(....)
     *
     * ticktock=gg.getTimes()
     *
     * ticktock.display()
    */
    GaussianProcessTimer& getTimes() const;

 private:
    int _pts,_useMaxMin,_dimensions,_room,_roomStep,_nFunctions;

    T _krigingParameter,_lambda;

    ndarray::Array<T,1,1> _max,_min;
    ndarray::Array<T,2,2> _function;

    KdTree<T> _kdTree;

    boost::shared_ptr< Covariogram<T> > _covariogram;
    mutable GaussianProcessTimer _timer;

};

}}}



#endif //#ifndef LSST_AFW_MATH_GAUSSIAN_PROCESS_H
