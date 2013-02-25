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
#include <time.h>
#include "ndarray/eigen.h"

namespace lsst{
namespace afw{
namespace math{

namespace GaussianProcessFunctions{

template <typename dty>
double euclideanDistance(dty*,dty*,int);

template <typename dtyi, typename dtyo>
dtyo expCovariogram(dtyi*,dtyi*,int,double*);

template <typename dtyi, typename dtyo>
dtyo neuralNetCovariogram(dtyi*,dtyi*,int,double*);

template<typename datatype>
void mergeSort(datatype*,int*,int);

template<typename datatype>
int mergeScanner(datatype*,int*,int,int);

}

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

template <typename datatype>
class KdTree{


 public:
    datatype **data;
 
    ~KdTree();
    KdTree(int,int,datatype**,double(*)(datatype*,datatype*,int));
    void findNeighbors(datatype*,int,int*,double*);
    void addPoint(datatype*);

    int getPoints();
    void getTreeNode(int,int*);
    int testTree();
     
   private:
    int **_tree,_pts,_dimensions,_room,_roomStep,*_inn,_masterParent;
    int *_neighborCandidates,_neighborsFound,_neighborsWanted;
    datatype *_toSort;
    double *_neighborDistances;
    
    void _organize(int*,int,int,int);
    int _findNode(datatype*);
    void _lookForNeighbors(datatype*,int,int);
    int _walkUpTree(int,int,int);
    
    double (*_distance)(datatype*,datatype*,int);
     
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

template <typename dtyi, typename dtyo>
class GaussianProcess{

  public:
  
     double interpolationTime,neighborSearchTime,inversionTime;
     double iterationTime,varSolveTime;
     int interpolationCount;
     
     enum{squaredExp,neuralNetwork};
    
     ~GaussianProcess();
     
     GaussianProcess(int,int,ndarray::Array<dtyi,2,2>,ndarray::Array<dtyo,1,1>);
      
     GaussianProcess(int,int,\
     ndarray::Array<dtyi,2,2>,ndarray::Array<dtyi,1,1>,ndarray::Array<dtyi,1,1>,ndarray::Array<dtyo,1,1>);
     
     
     //note: code will remember whether or not you input with maxs and
     //mins
     
     dtyo interpolate(ndarray::Array<dtyi,1,1>,ndarray::Array<dtyo,1,1>,int);
     dtyo selfInterpolate(int,ndarray::Array<dtyo,1,1>,int);

    
     void addPoint(ndarray::Array<dtyi,1,1>,dtyo);
     void setKrigingParameter(dtyo);
     void getNeighbors(ndarray::Array<int,1,1>);
     void setLambda(dtyo);
     void setHyperParameters(ndarray::Array<double,1,1>);
     void getCovarianceRow(int,ndarray::Array<dtyo,1,1>);
     int testKdTree();
     void getTimes();
     void resetTimes();
     void setCovariogramType(int);
     
     void batchInterpolate(ndarray::Array<dtyi,2,2>,ndarray::Array<dtyo,1,1>,ndarray::Array<dtyo,1,1>,int);
     void batchInterpolate(ndarray::Array<dtyi,2,2>,ndarray::Array<dtyo,1,1>,int);

  private:
    int _pts,_numberOfNeighbors,_useMaxMin,_dimensions,_room,_roomStep;
    int _calledInterpolate,*_neighbors,_nHyperParameters,_typeOfCovariogram;
    
    typedef Eigen::Matrix<dtyo,Eigen::Dynamic,Eigen::Dynamic> matrixtype;
    
    double *_neighborDistances,*_hyperParameters;
    dtyo *_function,*_covarianceTestPoint,_krigingParameter,_lambda;
    dtyi **_data,*_max,*_min,*_vv;
    
    Eigen::Matrix <dtyo,Eigen::Dynamic,Eigen::Dynamic> _covariance,_covarianceInverse;
    Eigen::Matrix <dtyo,Eigen::Dynamic,Eigen::Dynamic> _bb,_xx;
    
    Eigen::LLT<matrixtype> _llt;
    
    KdTree<dtyi> *_kdTreePtr;
          
    double (*_distance)(dtyi*,dtyi*,int);
    dtyo (*_covariogram)(dtyi*,dtyi*,int,double*);

};

}}}
