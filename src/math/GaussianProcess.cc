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
 * @file GaussianProcess.cc
 *
 * @ingroup afw
 *
 * @author Scott Daniel
 * Contact: scott.f.daniel@gmail.com
*/

#include "lsst/afw/math/GaussianProcess.h"
//#include "gptest/gptest.h"
#include "lsst/afw/math/detail/GaussianProcessFunctions.h"
#include <iostream>
#include <cmath>

using namespace std;

namespace lsst {
namespace afw {
namespace math {

namespace GPfn = lsst::afw::math::detail::GaussianProcess;

template <typename T>
KdTree<T>::~KdTree(){}

template <typename T>
KdTree<T>::KdTree(int dd, int pp, ndarray::Array<T,2,2> const &dt, \
double(*dfn)(ndarray::Array<T,1,1> const &,ndarray::Array<T,1,1> const &,int)){
  

  int i;
  
  //buffers to use when first building the tree
  _toSort=allocate(ndarray::makeVector(pp));
  _inn=allocate(ndarray::makeVector(pp));
 
  _dimensions=dd;
  _pts=pp;
  _roomStep=5000;
  _room=_pts;
  _distance=dfn;
  
  data=allocate(ndarray::makeVector(_room,_dimensions));
  
  data.deep()=dt;
  
  _tree=allocate(ndarray::makeVector(_room,4));
 
  for(i=0;i<_pts;i++){
    _inn[i]=i;
  }
  
  _organize(_inn,_pts,-1,-1);
  
}

template<typename T>
void KdTree<T>::findNeighbors(ndarray::Array<T,1,1> const &v, int n_nn, ndarray::Array<int,1,1> neighdex, \
ndarray::Array<double,1,1> dd){

  
  int i,start;
 
  ndarray::Array<int,1,1> order;
  ndarray::Array<double,1,1> dorder;  
  
  order=allocate(ndarray::makeVector(3));
  dorder=allocate(ndarray::makeVector(3));
  
  _neighborCandidates=allocate(ndarray::makeVector(n_nn));
  _neighborDistances=allocate(ndarray::makeVector(n_nn));
  _neighborsFound=0;
  _neighborsWanted=n_nn;
  
  for(i=0;i<n_nn;i++)_neighborDistances[i]=-1.0;
  
  start=_findNode(v);

  _neighborDistances[0]=_distance(v,data[start],_dimensions);
  _neighborCandidates[0]=start;
  _neighborsFound=1;
 
  
  if(_tree[start][3]>=0){
    dorder[2]=_distance(v,data[_tree[start][3]],_dimensions);
  
  }
  else dorder[2]=-1.0;
  order[2]=3;
  
  if(_tree[start][1]>=0){
    dorder[0]=_distance(v,data[_tree[start][1]],_dimensions);
  
  }
  else dorder[0]=-1.0;
  order[0]=1;
  
  if(_tree[start][2]>=0){
    dorder[1]=_distance(v,data[_tree[start][2]],_dimensions);
  
  }
  else dorder[1]=-1.0;
  order[1]=2;
  
  GPfn::mergeSort<double>(dorder,order,3);
  
  //search the branches in ascending order of distance from the test point
  //the idea being that if we look at branches that are closer first, we will
  //be more likely to rule out points quicker, speeding the search
  for(i=0;i<3;i++){
    if(_tree[start][order[i]]>=0){
      _lookForNeighbors(v,_tree[start][order[i]],start);
    }
  }
  
  for(i=0;i<n_nn;i++){
    neighdex[i]=_neighborCandidates[i];
    dd[i]=_neighborDistances[i];
  }
  
 
  
}

template <typename T>
void KdTree<T>::addPoint(ndarray::Array<T,1,1> const &v){

  int i,j,node,dim;
  
  node=_findNode(v);
  dim=_tree[node][0]+1;
  if(dim==_dimensions)dim=0;
  
  if(_pts==_room){
    
    ndarray::Array<T,2,2> dbuff=allocate(ndarray::makeVector(_pts,_dimensions));
  
    
    ndarray::Array<int,2,2> tbuff=allocate(ndarray::makeVector(_pts,4));
  
   
   dbuff.deep()=data;
   tbuff.deep()=_tree;

    _room+=_roomStep;
   
    _tree=allocate(ndarray::makeVector(_room,4)); 
    data=allocate(ndarray::makeVector(_room,_dimensions));
    
    
    for(i=0;i<_pts;i++){
      for(j=0;j<_dimensions;j++)data[i][j]=dbuff[i][j];
      for(j=0;j<4;j++)_tree[i][j]=tbuff[i][j];
   
    }

  }
  
  _tree[_pts][0]=dim;
  _tree[_pts][3]=node;
  i=_tree[node][0];
  
  if(data[node][i]>v[i]){
    if(_tree[node][1]>=0){
      std::cout<<"WARNING adding to a piece of tree that already exists 1\n";
      std::cout<<"node "<<node<<" "<<_tree[node][1]<<" "<<data[node][i]<<" "<<v[i]<<"\n";
      std::cout<<"pts "<<_pts<<"\n";
    }
    _tree[node][1]=_pts;
  }
  else{
    if(_tree[node][2]>=0){
      std::cout<<"WARNING adding to a piece of tree that already exists 2\n";
      std::cout<<"node "<<node<<" "<<_tree[node][2]<<" "<<data[node][i]<<" "<<v[i]<<"\n"; 
      std::cout<<"pts "<<_pts<<"\n";
    }
    _tree[node][2]=_pts;
  }
  _tree[_pts][1]=-1;
  _tree[_pts][2]=-1;
  for(i=0;i<_dimensions;i++){
    data[_pts][i]=v[i];
  }
  
  _pts++;
  
}

template <typename T>
int KdTree<T>::getPoints(){
  return _pts;
}

template <typename T>
void KdTree<T>::getTreeNode(int dex, ndarray::Array<int,1,1> v){
  v[0]=_tree[dex][0];
  v[1]=_tree[dex][1];
  v[2]=_tree[dex][2];
  v[3]=_tree[dex][3];
}

template <typename T>
int KdTree<T>::testTree(){

  int i,j,*isparent,output;
  
  j=0;
  for(i=0;i<_pts;i++){
    if(_tree[i][3]<0)j++;
  }
  if(j!=1){
    std::cout<<"_tree FAILURE "<<j<<" _masterParents\n";
    return 0;
  }
  
  isparent=new int[_pts];
  for(i=0;i<_pts;i++)isparent[i]=0;
  for(i=0;i<_pts;i++){
    isparent[_tree[i][3]]++;
  }
  for(i=0;i<_pts;i++){
    if(isparent[i]>2){ 
      std::cout<<"_tree FAILURE "<<i<<" is parent to "<<isparent[i]<<"\n";
      return 0;
    }
  }
  
  delete [] isparent;
  
  for(i=0;i<_pts;i++){
    
    if(_tree[i][3]>=0){
      if(_tree[_tree[i][3]][1]==i)j=1;
      else j=2;

      output=_walkUpTree(_tree[i][3],j,i);
      if(output!=_masterParent)return 0;
    }
  
  }
  std::cout<<"done with test of KdTree\n";
  if(output!=_masterParent) return 0;
  else return 1;
}

template <typename T>
void KdTree<T>::_organize(ndarray::Array<int,1,1> const &use, int ct, int parent, int dir){
    
  int i,j,k,l,idim,daughter;
  T mean,var,varbest;
   
  if(ct>1){
  //below is code to choose the dimension on which the available points
  //have the greates variance.  This will be the dimension on which
  //the daughter node splits the data
    for(i=0;i<_dimensions;i++){
      mean=0.0;
      var=0.0;
      for(j=0;j<ct;j++){
        mean+=data[use[j]][i];
        var+=data[use[j]][i]*data[use[j]][i];
      }
      mean=mean/double(ct);
      var=var/double(ct)-mean*mean;
      if(i==0 || var>varbest || (var==varbest && parent>=0 && i>_tree[parent][0])){
        idim=i;
        varbest=var;
      }
    
    }//for(i=0;i<_dimensions;i++)
  
    for(i=0;i<ct;i++){
      _toSort[i]=data[use[i]][idim];
    }
  
    GPfn::mergeSort<T>(_toSort,use,ct);
    
    k=ct/2;
    l=ct/2;
    while(k>0 && _toSort[k]==_toSort[k-1])k--;
   
    while(l<ct-1 && _toSort[l]==_toSort[ct/2])l++;
 
    if((ct/2-k)<(l-ct/2) || l==ct-1)j=k;
    else j=l;;
    
    daughter=use[j];

    if(parent>=0)_tree[parent][dir]=daughter;
    _tree[daughter][0]=idim;
    _tree[daughter][3]=parent;

    if(j<ct-1){
      //_organize(&use[j+1],ct-j-1,daughter,2);
      _organize(use[ndarray::view(j+1,use.getSize<0>())],ct-j-1,daughter,2);
    }
    else _tree[daughter][2]=-1;
  
    if(j>0){
      _organize(use,j,daughter,1);
    }
    else _tree[daughter][1]=-1;
    
  }//if(ct>1)
  else{
    daughter=use[0];
    if(parent>=0)_tree[parent][dir]=daughter;
    idim=_tree[parent][0]+1;
    if(idim>=_dimensions)idim=0;
    _tree[daughter][0]=idim;
    _tree[daughter][1]=-1;
    _tree[daughter][2]=-1;
    _tree[daughter][3]=parent;
    
  }
  
  if(parent==-1){
    _masterParent=daughter;
  }
  
}

template <typename T>
int KdTree<T>::_findNode(ndarray::Array<T,1,1> const &v){
  
  int consider,next,dim;
  
  dim=_tree[_masterParent][0];
  
  if(v[dim]<data[_masterParent][dim])consider=_tree[_masterParent][1];
  else consider=_tree[_masterParent][2];
  
  next=consider;
  
  while(next>=0){
    
    consider=next;
    
    dim=_tree[consider][0];
    if(v[dim]<data[consider][dim])next=_tree[consider][1];
    else next=_tree[consider][2];
  
  }
  
  return consider;
  
}

template<typename T>
void KdTree<T>::_lookForNeighbors(ndarray::Array<T,1,1> const &v, int consider, int from){

  int i,j,going;
  double dd;

  dd=_distance(v,data[consider],_dimensions);
  
  if(_neighborsFound<_neighborsWanted || dd<_neighborDistances[_neighborsWanted-1]){
    for(j=0;j<_neighborsFound && _neighborDistances[j]<dd;j++);
      
    for(i=_neighborsWanted-1;i>j;i--){
      _neighborDistances[i]=_neighborDistances[i-1];
      _neighborCandidates[i]=_neighborCandidates[i-1];
    }
    
    _neighborDistances[j]=dd;
    _neighborCandidates[j]=consider;
    
    if(_neighborsFound<_neighborsWanted)_neighborsFound++;
  }
  
  if(_tree[consider][3]==from){
    //you came here from the parent
    
    i=_tree[consider][0];
    dd=v[i]-data[consider][i];
    if((dd<=_neighborDistances[_neighborsFound-1] || _neighborsFound<_neighborsWanted) \
    && _tree[consider][1]>=0){
      _lookForNeighbors(v,_tree[consider][1],consider);
    }
    
    dd=data[consider][i]-v[i];
    if((dd<=_neighborDistances[_neighborsFound-1] || _neighborsFound<_neighborsWanted) \
    && _tree[consider][2]>=0){
      _lookForNeighbors(v,_tree[consider][2],consider);
    }
  }
  else{
    //you came here from one of the branches
    
    //descend the other branch
    if(_tree[consider][1]==from){
      going=2;
    }
    else{ 
      going=1;
    }
    
    j=_tree[consider][going];
    
    if(j>=0){
      i=_tree[consider][0];
      if(going==1)dd=v[i]-data[consider][i];
      else dd=data[consider][i]-v[i];
      
      if(dd<=_neighborDistances[_neighborsFound-1] || _neighborsFound<_neighborsWanted){
        _lookForNeighbors(v,j,consider);
      }
    }
    
    //ascend to the parent
    if(_tree[consider][3]>=0){
      
      _lookForNeighbors(v,_tree[consider][3],consider);
      
    }
    
  }
 

}

template <typename T>
int KdTree<T>::_walkUpTree(int target, int dir, int root){
  //target is the node that you are examining now
  //dir is where you came from
  //root is the ultimate point from which you started
  
  int i,output;
  
  output=1;
  
  if(dir==1){
    if(data[root][_tree[target][0]]>=data[target][_tree[target][0]]){
      std::cout<<"_tree FAILURE root "<<root<<" target "<<target<<" dir "<<dir<<"\n";
      std::cout<<data[root][_tree[target][0]]<<" >= "<<data[target][_tree[target][0]]<<"\n";
      output=0;
      return 0;
      
    }
  }
  else{
      if(data[root][_tree[target][0]]<data[target][_tree[target][0]]){
      
      std::cout<<"_tree FAILURE root "<<root<<"\n";
      std::cout<<" target "<<target<<" dir "<<dir<<" \n";
      std::cout<<data[root][_tree[target][0]]<<" < "<<data[target][_tree[target][0]]<<"\n";
      output=0;
      return 0;

    }
  }
  
  if(_tree[target][3]>=0){
    if(_tree[_tree[target][3]][1]==target)i=1;
    else i=2;
    
    output=output*_walkUpTree(_tree[target][3],i,root);
  
  }
  else{
    output=output*target;
    //so that it will return _masterParent
    //make sure everything is connected to _masterParent
  }
  return output;
  
}


template <typename T>
GaussianProcess<T>::~GaussianProcess(){
    delete _kdTreePtr;
}

template <typename T>
GaussianProcess<T>::GaussianProcess(int dd, int pp, ndarray::Array<T,2,2> const &datain, \
ndarray::Array<T,1,1> const &ff){
 

    
  int i,j;
  ndarray::Array<int,2,2> ndtest;
  
  ndtest=allocate(ndarray::makeVector(3,3));
  
  _dimensions=dd;
  _pts=pp;
  _room=_pts;
  _roomStep=5000;
  
  _function=allocate(ndarray::makeVector(_pts));
  _function.deep()=ff;
  _krigingParameter=T(1.0);
  
  _covariogram=GPfn::expCovariogram;
  _distance=GPfn::euclideanDistance;
  
  _calledInterpolate=0;
  
  _lambda=T(1.0e-5);
  
  _useMaxMin=0;
  
  
  _data=allocate(ndarray::makeVector(_pts,_dimensions));
  
  for(i=0;i<_pts;i++){
     for(j=0;j<_dimensions;j++){
       _data[i][j]=datain[i][j];
     }
  }
  
  _kdTreePtr=new KdTree<T>(_dimensions,_pts,_data,_distance);
  

  
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();
  
  _typeOfCovariogram=squaredExp;
  _nHyperParameters=1;
  _hyperParameters=allocate(ndarray::makeVector(1));
  _hyperParameters[0]=1.0;
  
  interpolationTime=0.0;
  interpolationCount=0;
  neighborSearchTime=0.0;
  inversionTime=0.0;
  iterationTime=0.0;
  varSolveTime=0.0;
  
}

template <typename T>
GaussianProcess<T>::GaussianProcess(int dd, int pp, ndarray::Array<T,2,2> const &datain,\
ndarray::Array<T,1,1> const &mn, ndarray::Array<T,1,1> const &mx, ndarray::Array<T,1,1> const &ff){

  int i,j;
  
  _dimensions=dd;
  _pts=pp;
  _room=_pts;
  _roomStep=5000;
  
  _krigingParameter=T(1.0);
    
  _covariogram=GPfn::expCovariogram;
  _distance=GPfn::euclideanDistance;
  
  _calledInterpolate=0;

 _lambda=T(1.0e-5);
 _krigingParameter=T(1.0);
   
   _max=allocate(ndarray::makeVector(_dimensions));
   _min=allocate(ndarray::makeVector(_dimensions));
   
 _max.deep()=mx;
  _min.deep()=mn;
  
 _useMaxMin=1;
 
 _data=allocate(ndarray::makeVector(_pts,_dimensions));
  
  
  for(i=0;i<_pts;i++){
   
    for(j=0;j<_dimensions;j++){
      _data[i][j]=(datain[i][j]-_min[j])/(_max[j]-_min[j]); //note the normalization by _max-_min in each dimension
    }
  }
   
  _kdTreePtr=new KdTree<T>(_dimensions,_pts,_data,_distance);
  
 
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();
  
  _function=allocate(ndarray::makeVector(_pts));
  _function.deep()=ff;
  
  _typeOfCovariogram=squaredExp;
  _nHyperParameters=1;
  _hyperParameters=allocate(ndarray::makeVector(1));
  _hyperParameters[0]=1.0;
  
  interpolationTime=0.0;
  interpolationCount=0;
  neighborSearchTime=0.0;
  inversionTime=0.0;
  iterationTime=0.0;
  varSolveTime=0.0;
  

}

template <typename T>
T GaussianProcess<T>::interpolate(ndarray::Array<T,1,1> const &vin, ndarray::Array<T,1,1> variance, int kk){

  int i,j;
  T fbar,mu;
  double before,after,aa,bb;
  
  
  before=double(::time(NULL));
  
  if(_calledInterpolate==0 || kk!=_numberOfNeighbors){
  //if this is not the first time you have called this method, the code must make sure that the
  //arrays it uses are large enough to accommodate the number of nearest neighbors you asked for
  
     _covarianceTestPoint=allocate(ndarray::makeVector(kk));
     _covariance.resize(kk,kk);
     _bb.resize(kk,1);
     _xx.resize(kk,1);
     
     _neighbors=allocate(ndarray::makeVector(kk));;
     _neighborDistances=allocate(ndarray::makeVector(kk));
     
     _numberOfNeighbors=kk;
  }
  
  if(_calledInterpolate==0){
    _vv=allocate(ndarray::makeVector(_dimensions));
  }
  
  if(_useMaxMin==1){
    //if you constructed this Gaussian process with minimum and maximum values for the dimensions of your parameter space,
    //the point you are interpolating must be scaled to match the data so that the selected nearest neighbors are appropriate
    
    for(i=0;i<_dimensions;i++)_vv[i]=(vin[i]-_min[i])/(_max[i]-_min[i]);
  }
  else{
    /*for(i=0;i<_dimensions;i++){
      _vv[i]=vin[i];
    }*/
    _vv=vin;
  }
  
  bb=double(::time(NULL));
  _kdTreePtr->findNeighbors(_vv,_numberOfNeighbors,_neighbors,_neighborDistances);
  aa=double(::time(NULL));
  
  neighborSearchTime+=aa-bb;
  
  bb=double(::time(NULL));
  fbar=0.0;
  for(i=0;i<_numberOfNeighbors;i++)fbar+=_function[_neighbors[i]];
  fbar=fbar/double(_numberOfNeighbors);

  for(i=0;i<_numberOfNeighbors;i++){
    _covarianceTestPoint[i]=_covariogram(_vv,_data[_neighbors[i]],_dimensions,_hyperParameters);
    _covariance(i,i)=_covariogram(_data[_neighbors[i]],_data[_neighbors[i]],_dimensions,_hyperParameters)\
    +_lambda;
    for(j=i+1;j<_numberOfNeighbors;j++){
      _covariance(i,j)=_covariogram(_data[_neighbors[i]],_data[_neighbors[j]],_dimensions,_hyperParameters);
      _covariance(j,i)=_covariance(i,j);
    }
  }
  
  aa=double(::time(NULL));
  iterationTime+=aa-bb;

  bb=double(::time(NULL));
  
  //use Eigen's llt solver in place of matrix inversion (for speed purposes)
  _llt.compute(_covariance); 
  
  for(i=0;i<_numberOfNeighbors;i++)_bb(i,0)=_function[_neighbors[i]]-fbar;
  _xx=_llt.solve(_bb);
  aa=double(::time(NULL));
  
  inversionTime+=aa-bb;
  
  bb=double(::time(NULL));
  mu=fbar;

  for(i=0;i<_numberOfNeighbors;i++){
    mu+=_covarianceTestPoint[i]*_xx(i,0);
  }
  
  variance(0)=_covariogram(_vv,_vv,_dimensions,_hyperParameters)+_lambda;
  
  for(i=0;i<_numberOfNeighbors;i++)_bb(i)=_covarianceTestPoint[i];

  _xx=_llt.solve(_bb);
  aa=double(::time(NULL));
  varSolveTime+=aa-bb;
  
  bb=double(::time(NULL));
  for(i=0;i<_numberOfNeighbors;i++){
    variance(0)-=_covarianceTestPoint[i]*_xx(i,0);
  } 
  aa=double(::time(NULL));
  iterationTime+=aa-bb;
  
  variance(0)=variance(0)*_krigingParameter;
  
  _calledInterpolate=1;
  
  after=double(::time(NULL));
  interpolationTime+=after-before;
  interpolationCount++;
  

  return mu;
}

template <typename T>
T GaussianProcess<T>::selfInterpolate(int dex, ndarray::Array<T,1,1> variance, int kk){
  
  int i,j;
  T fbar,mu;
  double before,after,aa,bb;

  
  ndarray::Array<int,1,1> selfNeighbors;
  ndarray::Array<double,1,1> selfDistances;
  
  before=double(::time(NULL));
  
  if(_calledInterpolate==0 || kk!=_numberOfNeighbors){
  //if this is not the first time you have called this method, the code must make sure that the
  //arrays it uses are large enough to accommodate the number of nearest neighbors you asked for
  
     _covarianceTestPoint=allocate(ndarray::makeVector(kk));
     _covariance.resize(kk,kk);
     _bb.resize(kk,1);
     _xx.resize(kk,1);
     
     _neighbors=allocate(ndarray::makeVector(kk));
     _neighborDistances=allocate(ndarray::makeVector(kk));
     
     _numberOfNeighbors=kk;
  }
  
  selfNeighbors=allocate(ndarray::makeVector(_numberOfNeighbors+1));
  selfDistances=allocate(ndarray::makeVector(_numberOfNeighbors+1));
  
  if(_calledInterpolate==0){
    _vv=allocate(ndarray::makeVector(_dimensions));
  }
  
  //we don't use _useMaxMin because _data has already been normalized
    for(i=0;i<_dimensions;i++){
      _vv[i]=_data[dex][i];
    }
  
  
  bb=double(::time(NULL));
  _kdTreePtr->findNeighbors(_vv,_numberOfNeighbors+1,selfNeighbors,selfDistances);

  
  if(selfNeighbors[0]!=dex){
    std::cout<<"WARNING selfdist "<<selfDistances[0]<<" "<<selfDistances[1]<<"\n";
    std::cout<<"dex "<<dex<<" "<<selfNeighbors[0]<<"\n";
    exit(1);
  }
  
  //SelfNeighbors[0] will be the point itself (it is its own nearest neighbor)
  //We discard that for the interpolation calculation
  //
  //If you do not wish to do this, simply call the usual ::interpolate() method instead of
  //::selfInterpolate()
  for(i=0;i<_numberOfNeighbors;i++){
    _neighbors[i]=selfNeighbors[i+1];
    _neighborDistances[i]=selfDistances[i+1];
  }
  aa=double(::time(NULL));
  neighborSearchTime+=aa-bb;
  
  bb=double(::time(NULL));
  fbar=0.0;
  for(i=0;i<_numberOfNeighbors;i++)fbar+=_function[_neighbors[i]];
  fbar=fbar/double(_numberOfNeighbors);

  for(i=0;i<_numberOfNeighbors;i++){
    _covarianceTestPoint[i]=_covariogram(_vv,_data[_neighbors[i]],_dimensions,_hyperParameters);
    _covariance(i,i)=_covariogram(_data[_neighbors[i]],_data[_neighbors[i]],_dimensions,_hyperParameters)\
    +_lambda;
    for(j=i+1;j<_numberOfNeighbors;j++){
      _covariance(i,j)=_covariogram(_data[_neighbors[i]],_data[_neighbors[j]],_dimensions,_hyperParameters);
      _covariance(j,i)=_covariance(i,j);
    }
  }
  
  aa=double(::time(NULL));
  iterationTime+=aa-bb;

  bb=double(::time(NULL));
  
  //use Eigen's llt solver in place of matrix inversion (for speed purposes)
  _llt.compute(_covariance); 
  
  
  for(i=0;i<_numberOfNeighbors;i++)_bb(i,0)=_function[_neighbors[i]]-fbar;
  _xx=_llt.solve(_bb);
  aa=double(::time(NULL));
  
  inversionTime+=aa-bb;
  
  
  bb=double(::time(NULL));
  mu=fbar;

  for(i=0;i<_numberOfNeighbors;i++){
    mu+=_covarianceTestPoint[i]*_xx(i,0);
  }
  
  variance(0)=_covariogram(_vv,_vv,_dimensions,_hyperParameters)+_lambda;
  
  for(i=0;i<_numberOfNeighbors;i++)_bb(i)=_covarianceTestPoint[i];
  aa=double(::time(NULL));
  iterationTime+=aa-bb;
  
  bb=double(::time(NULL));
  _xx=_llt.solve(_bb);
  aa=double(::time(NULL));
  varSolveTime+=aa-bb;
  
  bb=double(::time(NULL));
  for(i=0;i<_numberOfNeighbors;i++){
    variance(0)-=_covarianceTestPoint[i]*_xx(i,0);
  } 
  aa=double(::time(NULL));
  iterationTime+=aa-bb;
  
  variance(0)=variance(0)*_krigingParameter;
  
  _calledInterpolate=1;
  
  after=double(::time(NULL));
  interpolationTime+=after-before;
  interpolationCount++;
  
  return mu;
}

template<typename T>
void GaussianProcess<T>::batchInterpolate(ndarray::Array<T,2,2> const &queries, ndarray::Array<T,1,1> mu, \
ndarray:: Array<T,1,1> variance, int nQueries){
    
  int i,j,ii;
  double aa,bb,before;
  T fbar;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  
  ndarray::Array<T,1,1> v1; 

  bb=double(::time(NULL));
  before=bb;
  
  v1=allocate(ndarray::makeVector(_dimensions));
  batchbb.resize(_pts,1);
  batchxx.resize(_pts,1);
  batchCovariance.resize(_pts,_pts);
  queryCovariance.resize(_pts,1);
 
  
  for(i=0;i<_pts;i++){
    
    batchCovariance(i,i)=_covariogram(_data[i],_data[i],_dimensions,_hyperParameters)+_lambda;
    for(j=i+1;j<_pts;j++){
      batchCovariance(i,j)=_covariogram(_data[i],_data[j],_dimensions,_hyperParameters);
      batchCovariance(j,i)=batchCovariance(i,j);
    }
  }
  
  _llt.compute(batchCovariance);  
  
  fbar=0.0;
  for(i=0;i<_pts;i++){
    fbar+=_function[i];
  }
  fbar=fbar/T(_pts);
  
  //std::cout<<"fbar "<<fbar<<"\n";
  
  for(i=0;i<_pts;i++){
    batchbb(i,0)=_function[i]-fbar;
  }
  batchxx=_llt.solve(batchbb);
  aa=double(::time(NULL));
  inversionTime+=aa-bb;
  
  
  for(ii=0;ii<nQueries;ii++){
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    } 
    mu(ii)=fbar;
    for(i=0;i<_pts;i++){
      mu(ii)+=batchxx(i)*_covariogram(v1,_data[i],_dimensions,_hyperParameters);
    /* if(ii==0){
       std::cout<<"mu "<<mu(ii)<<" xx "<<batchxx(i)<<" cov "<<_covariogram(v1,_data[i],_dimensions,_hyperParameters)<<"\n";
     }*/
    }
  }
  bb=double(::time(NULL));
  iterationTime+=bb-aa;

  
  //std::cout<<"done with interpolation\n";
  
  bb=double(::time(NULL));
  for(ii=0;ii<nQueries;ii++){
    //std::cout<<"i "<<ii<<"\n";
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    }
    
    for(i=0;i<_pts;i++){
      batchbb(i,0)=_covariogram(v1,_data[i],_dimensions,_hyperParameters);
      queryCovariance(i,0)=batchbb(i,0);
    }
    batchxx=_llt.solve(batchbb);
    
    variance(ii)=_covariogram(v1,v1,_dimensions,_hyperParameters)+_lambda;
    
    for(i=0;i<_pts;i++){
      variance(ii)-=queryCovariance(i,0)*batchxx(i);
    }
    
    variance(ii)=variance(ii)*_krigingParameter;
      
  }
  aa=double(::time(NULL));
  varSolveTime+=aa-bb;
  interpolationTime+=aa-before;
  interpolationCount+=nQueries;

}

template<typename T>
void GaussianProcess<T>::batchInterpolate(ndarray::Array<T,2,2> const &queries, ndarray::Array<T,1,1> mu,\
 int nQueries){

  int i,j,ii;
  double aa,bb,before,after;

  T fbar;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <T,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  ndarray::Array<T,1,1> v1;
  
  bb=double(::time(NULL));
  before=bb;
  
  v1=allocate(ndarray::makeVector(_dimensions));
  
  batchbb.resize(_pts,1);
  batchxx.resize(_pts,1);
  batchCovariance.resize(_pts,_pts);
  queryCovariance.resize(_pts,1);
 
  
  for(i=0;i<_pts;i++){
    batchCovariance(i,i)=_covariogram(_data[i],_data[i],_dimensions,_hyperParameters)+_lambda;
    for(j=i+1;j<_pts;j++){
      batchCovariance(i,j)=_covariogram(_data[i],_data[j],_dimensions,_hyperParameters);
      batchCovariance(j,i)=batchCovariance(i,j);
    }
  }
  
  _llt.compute(batchCovariance);  

  fbar=0.0;
  for(i=0;i<_pts;i++){
    fbar+=_function[i];
  }
  fbar=fbar/T(_pts);
  
  //std::cout<<"fbar "<<fbar<<"\n";
  
  for(i=0;i<_pts;i++){
    batchbb(i,0)=_function[i]-fbar;
  }
  batchxx=_llt.solve(batchbb);
  aa=double(::time(NULL));
  inversionTime+=aa-bb;
  
  
  for(ii=0;ii<nQueries;ii++){
    for(i=0;i<_dimensions;i++)v1[i]=queries(ii,i);
    if(_useMaxMin==1){
      for(i=0;i<_dimensions;i++)v1[i]=(v1[i]-_min[i])/(_max[i]-_min[i]);
    }
    
    mu(ii)=fbar;
    for(i=0;i<_pts;i++){
      mu(ii)+=batchxx(i)*_covariogram(v1,_data[i],_dimensions,_hyperParameters);
    /* if(ii==0){
       std::cout<<"mu "<<mu(ii)<<" xx "<<batchxx(i)<<" cov "<<_covariogram(v1,_data[i],_dimensions,_hyperParameters)<<"\n";
     }*/
    }
  }
  after=double(::time(NULL));
  iterationTime+=after-aa;
  interpolationTime+=after-before;
  interpolationCount+=nQueries;
  
  //std::cout<<"done with interpolation\n";
}

template <typename T>
void GaussianProcess<T>::addPoint(ndarray::Array<T,1,1> const &vin, T f){

  int i;
  
  ndarray::Array<T,1,1> v;
  v=allocate(ndarray::makeVector(_dimensions));
 
  for(i=0;i<_dimensions;i++){
    v[i]=vin[i];
    if(_useMaxMin==1){
      v[i]=(v[i]-_min[i])/(_max[i]-_min[i]);
    }
    
  }
  
  if(_pts==_room){
    ndarray::Array<T,1,1> buff;
    buff=allocate(ndarray::makeVector(_pts));
    buff.deep()=_function;
    
    _room+=_roomStep;
    _function=allocate(ndarray::makeVector(_room));
    for(i=0;i<_pts;i++){
      _function[i]=buff[i];
    }
 
  }
  _function[_pts]=f;
  
  _kdTreePtr->addPoint(v);
  _pts=_kdTreePtr->getPoints();
  _data=_kdTreePtr->data;
  

}

template <typename T>
void GaussianProcess<T>::setKrigingParameter(T kk){
  
  _krigingParameter=kk;
  
}


template <typename T>
void GaussianProcess<T>::setLambda(T ll){

  _lambda=ll;

}

template <typename T>
void GaussianProcess<T>::setHyperParameters(ndarray::Array<double,1,1> const &hyin){

  int i;
  for(i=0;i<_nHyperParameters;i++){
    _hyperParameters[i]=hyin[i];
  }
  
}


template <typename T>
void GaussianProcess<T>::setCovariogramType(int ii){
  
  int i;
  
  switch(ii){
    case squaredExp:
      _nHyperParameters=1;
      _covariogram=GPfn::expCovariogram;
    break;
    case neuralNetwork:
     _nHyperParameters=2;
     _covariogram=GPfn::neuralNetCovariogram;
    break;
    default:
     std::cout<<"I do not know that kind; I will set the squared exponent\n";
     _nHyperParameters=1;
     _covariogram=GPfn::expCovariogram;
  
  }
  
  _hyperParameters=allocate(ndarray::makeVector(_nHyperParameters));
  for(i=0;i<_nHyperParameters;i++){
    _hyperParameters[i]=1.0;
  }
  
  _typeOfCovariogram=ii;
  
}

template <typename T>
void GaussianProcess<T>::getNeighbors(ndarray::Array<int,1,1> v){
  int i;
  if(_calledInterpolate==0){
    std::cout<<"You cannot call getNeighbors; you have not called interpolate at all\n";
    //printf("You cannot call print_nn; you haven't called interpolate at all\n");
  }
  else{
    for(i=0;i<_numberOfNeighbors;i++)v(i)=_neighbors[i];
  }
}

template <typename T>
void GaussianProcess<T>::getCovarianceRow(int dex, ndarray::Array<T,1,1> v){

  int i;
  if(_calledInterpolate==0){
    std::cout<<"You cannot call getCovarianceRow; you have not called interpolate\n";
    //printf("You can't call print gg_row; you haven't called interpolate\n");
  }
  else{
    for(i=0;i<_numberOfNeighbors;i++)v(i)=_covariance(dex,i);
  }
}


template <typename T>
int GaussianProcess<T>::testKdTree(){

  return _kdTreePtr->testTree();

}


template <typename T>
void GaussianProcess<T>::getTimes(){
  std::cout<<"\n";
  std::cout<<"interpolate time "<<interpolationTime<<"\n";
  std::cout<<"search time "<<neighborSearchTime<<"\n";
  std::cout<<"inversion time "<<inversionTime<<"\n";
  std::cout<<"var solve time "<<varSolveTime<<"\n";
  std::cout<<"iteration time "<<iterationTime<<"\n";
  std::cout<<"called interpolate "<<interpolationCount<<" times\n";
  
 // printf("interpolate time %.4e\n",interpolationTime);
 // printf("search time %.4e\n",neighborSearchTime);
  //printf("inversion time %.4e\n",inversionTime);
 // printf("iteration time %4e\n",iterationTime);
 // printf("var solve time %.4e\n",varSolveTime);
  
  std::cout<<"\n";
}

template <typename T>
void GaussianProcess<T>::resetTimes(){
  interpolationTime=0.0;
  neighborSearchTime=0.0;
  inversionTime=0.0;
  varSolveTime=0.0;
  iterationTime=0.0;
  interpolationCount=0;
}





template <typename T>
void GaussianProcess<T>::waste(ndarray::Array<T,2,2> const &aa){
  
  std::cout<<"in waste aa "<<aa[2][2]<<"\n";
  
}

}}}

#define gpn lsst::afw::math

#define INSTANTIATEGP(T) \
	template class gpn::GaussianProcess<T>;

INSTANTIATEGP(double);


