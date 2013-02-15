#include <Eigen/Dense>
#include <time.h>
#include "ndarray/eigen.h"

namespace gptest {

namespace GaussianProcessFunctions{

template <typename dty>
double euclideanDistance(dty*,dty*,int);

template <typename dtyi, typename dtyo>
dtyo expCovariogram(dtyi*,dtyi*,int);

template<typename datatype>
void mergeSort(datatype*,int*,int);

template<typename datatype>
int mergeScanner(datatype*,int*,int,int);

}

template <typename datatype>
class KdTree{
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

 public:
    datatype **data;
 
    ~KdTree();
    KdTree(int,int,datatype**,double(*)(datatype*,datatype*,int));
    void findNeighbors(datatype*,int,int*,double*);
    void addPoint(datatype*);

    int getPoints();
    void getTreeNode(int,int*);
    void testSort();
    void testScanner();
    int testTree();
     
};


template <typename dtyi, typename dtyo>
class GaussianProcess{

  private:
    int _pts,_numberOfNeighbors,_useMaxMin,_dimensions,_room,_roomStep;
    int _calledInterpolate,*_neighbors,_nHyperParameters;
    
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
      
  public:
  
     double interpolationTime,neighborSearchTime,inversionTime;
     double iterationTime,varSolveTime;
     int interpolationCount;
    
     ~GaussianProcess();
     
     GaussianProcess(int,int,ndarray::Array<dtyi,2,2>,ndarray::Array<dtyo,1,1>);
      
     /*GaussianProcess(int,int,\
     ndarray::Array<dtyi,2,2>,ndarray::Array<dtyi,2,2>,ndarray::Array<dtyi,2,2>,dtyo*);
     */
     
     //note: code will remember whether or not you input with maxs and
     //mins
     
     dtyo interpolate(ndarray::Array<dtyi,1,1>,ndarray::Array<dtyo,1,1>,int);
    
     void addPoint(ndarray::Array<dtyi,1,1>,dtyo);
     void setKrigingParameter(int);
     void getNeighbors(ndarray::Array<int,1,1>);
     void setLambda(dtyo);
     void setHyperParameters(ndarray::Array<double,1,1>);
     void getCovarianceRow(int,ndarray::Array<dtyo,1,1>);
     int testKdTree();
     void getTimes();

};

}
