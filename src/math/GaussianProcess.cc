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
#include <iostream>
#include <cmath>

using namespace std;

namespace lsst{
namespace afw{
namespace math{

namespace GaussianProcessFunctions{

/** 
 * This namespace contains some functions that need to be `global'
 * so that both GaussianProcess and KdTree can access them
*/

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
*/

template <typename datatype>
int mergeScanner(datatype *m, int *indices, int dex, int el){
  /**
    * @param m is a list of numbers to be sorted
    *
    * @param indices is a list of ints which keeps track of the sorted numbers original
    * positions
    * 
    * @param dex denotes the value about which everything is to be sorted
    * (i.e. values less than m[dex] will get put to the left of it;
    * values greater than m[dex] will get put to the right)
    *
    * @param el denotes how many elements are in m[] and indices[]
  */
  
  int i,j,k,newdex;
  datatype nn;
    
  newdex=0;
  for(i=0;i<el;i++){
    if(m[i]<m[dex])newdex++;
  }
   
  nn=m[newdex];
  j=indices[newdex];
  
  m[newdex]=m[dex];
  indices[newdex]=indices[dex];
  
  m[dex]=nn;
  indices[dex]=j;
  
  //now that m[dex] is in the right place
  
  i=0;
  j=el-1;
  
  while(i<newdex && j>newdex){
    if(m[i]<m[newdex] && m[j]>=m[newdex]){
      i++;
      j--;
    }
    else if(m[i]<m[newdex] && m[j]<m[newdex]){
      i++;
    }
    else if(m[i]>=m[newdex] && m[j]>=m[newdex]){
      j--;
    }
    else{
      
      nn=m[i];
      k=indices[i];
      
      m[i]=m[j];
      indices[i]=indices[j];
      
      m[j]=nn;
      indices[j]=k;
      
      i++;
      j--;
      
    }
  }//walking along m[] from either direction
  
  return newdex;
  
}

/**
 * @brief Sort a list of numbers using a merge sort algorithm
 *
 * mergeSort is the `outer' method which implements the merge sort
 * algorithm from Numerical Recipes.  It relies on mergeScanner
 * to be complete
*/

template <typename datatype>
void mergeSort(datatype *insort, int *indices, int el){
  
  /**
   * @param insort is the list of numbers to be sorted
   *
   * @param indices Keeps track of their original order (in case there is another
   * list that needs to be correlated with the list of sorted values)
   *
   * @param el is the number of values being sorted
  */
  
  int i,k;
  datatype nn;
  
  if(el==2){
    if(insort[0]>=insort[1]){
      nn=insort[0];
      k=indices[0];
      
      insort[0]=insort[1];
      indices[0]=indices[1];
      
      insort[1]=nn;
      indices[1]=k;
    }
  }
  else if(el>2){
    i=mergeScanner<datatype>(insort,indices,el/2,el);
  
    if(i>1){
      mergeSort<datatype>(insort,indices,i);
    }
    
    if(i<el-2){
      mergeSort<datatype>(&insort[i+1],&indices[i+1],el-i-1);
    }
  
  }

}

/**
 * @brief Return the Euclidean distance between two points in arbitrary-dimensional space
 *
 * This is left as a method in case future developers want to enable the possibility that
 * KdTree will define nearest neighbors by some other distance
*/

template<typename dty>
double euclideanDistance(dty *v1, dty *v2, int d_dim){
 /**
  * @param v1 the first point
  *
  * @param v2 the second point
  *
  * @param d_dim the number of dimensions in the parameter space
 */


  int i;
  double dd;
  dd=0.0;
  for(i=0;i<d_dim;i++){
    dd+=double(v1[i]-v2[i])*double(v1[i]-v2[i]);
  }
  
  return ::sqrt(dd);
}

/**
 * @brief The squared exponential covariogram for GaussianProcess
 *
 * Takes two points in parameter space and returns the covariogram relation
 * between them
*/

template<typename dtyi, typename dtyo>
dtyo expCovariogram(dtyi *v1, dtyi *v2, int d_dim, double *hyp){
 /**
  * @param v1 the first point
  *
  * @param v2 the second point
  *
  * @param d_dim the number of dimensions in parameter space
  *
  * @param hyp a list of hyperparameters governing the shape of the covariogram
  *
  * in this case, there is only one hyperparameter: the characteristic length scale squared
 */


  double dd;
  dd=euclideanDistance(v1,v2,d_dim);
  return dtyo(::exp(-0.5*dd*dd/hyp[0]));
}

/**
 * @brief The covariogram of a neural network with infinite hidden layers
 *
 * See Chapter 4 of Rasmussen and Williams (2006)
 * http://gaussianprocess.org/gpml/
 * equation (4.29)
*/
template<typename dtyi, typename dtyo>
dtyo neuralNetCovariogram(dtyi *v1, dtyi *v2, int d_dim, double *hyp){
 /**
  * @param v1 the first point
  *
  * @param v2 the second point
  *
  * @param d_dim the number of dimensions in parameter space
  *
  * @param hyp a list of hyperparameters governing the shape of the covariogram
  *
  * in this case, there are two hyper parameters as defined by Rasmussen and Williams
  * (they call them \sigma^2 and \sigma^2_0)
 */

  int i;
  double num,denom1,denom2,arg;
  
  num=2.0*hyp[0];
  denom1=1.0+2.0*hyp[0];
  denom2=1.0+2.0*hyp[0];
  for(i=0;i<d_dim;i++){
    num+=2.0*v1[i]*hyp[1]*v2[i];
    denom1+=2.0*v1[i]*hyp[1]*v1[i];
    denom2+=2.0*v2[i]*hyp[1]*v2[i];
  }
  arg=num/::sqrt(denom1*denom2);
  
  if(arg>1.0 || arg<-1.0){
    std::cout<<"WARNING in neural network covariogram "<<arg<<" cannot be outside of [-1,1]\n";
    exit(1);
  }
  
  return dtyo(2.0*(::asin(arg))/3.141592654);
  
  
}

}


namespace GPfn = lsst::afw::math::GaussianProcessFunctions;

template <typename datatype>
KdTree<datatype>::~KdTree(){
  int i;
  for(i=0;i<_room;i++){
    delete [] data[i];
    delete [] _tree[i];
  }
  delete [] data;
  delete [] _tree;
}

/**
 * @brief Build a KD Tree to store the data for GaussianProcess
*/

template <typename datatype>
KdTree<datatype>::KdTree(int dd, int pp, datatype **dt, \
double(*dfn)(datatype*,datatype*,int)){
  
  /**
   * @param dd the number of dimensions of parameter space
   *
   * @param pp the number of data points being read in
   *
   * @param dt an array, the rows of which are the data points (dt[i][j] is the jth component of the ith data point)
   *
   * @param dfn a function defining the distance by which KdTree will define ``nearest neighbors''
  */
  
  int i,j;
  
  //buffers to use when first building the tree
  _toSort=new datatype[pp];
  _inn=new int[pp];
  
 
  _dimensions=dd;
  _pts=pp;
  _roomStep=5000;
  _room=_pts;
  _distance=dfn;
  
  data=new datatype*[_room];
  for(i=0;i<_room;i++){
    data[i]=new datatype[_dimensions];
    for(j=0;j<_dimensions;j++)data[i][j]=dt[i][j];
  }
  
  
  _tree=new int*[_pts];
  for(i=0;i<_pts;i++){
    _inn[i]=i;
    _tree[i]=new int[4];
  }
  
  _organize(_inn,_pts,-1,-1);
  
  delete [] _toSort;
  delete [] _inn;
  
}

/**
 * @brief Find the daughter point of a node in the tree and segregate the points around it
*/

template <typename datatype>
void KdTree<datatype>::_organize(int *use, int ct, int parent, int dir){
  
  /**
   * @param use the indices of the data points being considered as possible daughters
   *
   * @param ct the number of possible daughters
   *
   * @param parent the index of the parent whose daughter we are chosing
   *
   * @param dir which side of the parent are we on?  dir==1 means that we are on the left side; dir==2 means the right side.
  */
  
  int i,j,k,l,idim,daughter;
  datatype mean,var,varbest;
  
  
  
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
  
    GPfn::mergeSort<datatype>(_toSort,use,ct);
    
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
      _organize(&use[j+1],ct-j-1,daughter,2);
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

/**
* @brief Return the _tree information for a given data point
*/
template <typename datatype>
void KdTree<datatype>::getTreeNode(int dex, int *v){
  v[0]=_tree[dex][0];
  v[1]=_tree[dex][1];
  v[2]=_tree[dex][2];
  v[3]=_tree[dex][3];
}

/**
 * @brief A method to make sure that every data point in the tree is in the correct relation to its parents
*/

template <typename datatype>
int KdTree<datatype>::_walkUpTree(int target, int dir, int root){
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

/**
 * @brief Make sure that the tree is properly constructed.  Returns 1 of it is.
*/
template <typename datatype>
int KdTree<datatype>::testTree(){

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

/**
 * @brief Find the point already in the tree that would be the parent of a point not in the tree
*/
template <typename datatype>
int KdTree<datatype>::_findNode(datatype *v){
  
  /**
   * @param v the points whose prospective parent you want to find
  */

  
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

/**
 * @brief Add a point to the tree.  Allot more space in _tree and data if needed.
*/
template <typename datatype>
void KdTree<datatype>::addPoint(datatype *v){
 
  /**
   * @param v the point you are adding to the tree
  */

  int i,j,**tbuff,node,dim;
  datatype **dbuff;
  
  node=_findNode(v);
  dim=_tree[node][0]+1;
  if(dim==_dimensions)dim=0;
  
  if(_pts==_room){
    tbuff=new int*[_room];
    dbuff=new datatype*[_room];
    
    for(i=0;i<_room;i++){
      tbuff[i]=new int[4];
      dbuff[i]=new datatype[_dimensions];
      for(j=0;j<_dimensions;j++)dbuff[i][j]=data[i][j];
      delete [] data[i];
      for(j=0;j<4;j++)tbuff[i][j]=_tree[i][j];
      delete [] _tree[i];
    }
    delete [] _tree;
    delete [] data;
    _room+=_roomStep;
    _tree=new int*[_room];
    data=new datatype*[_room];
    for(i=0;i<_room;i++){
      _tree[i]=new int[4];
      data[i]=new datatype[_dimensions];
    }
    
    for(i=0;i<_pts;i++){
      for(j=0;j<_dimensions;j++)data[i][j]=dbuff[i][j];
      delete [] dbuff[i];
      for(j=0;j<4;j++)_tree[i][j]=tbuff[i][j];
      delete [] tbuff[i];
    }
    delete [] dbuff;
    delete [] tbuff;
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

/**
 * @brief Find the nearest neighbors of a point
*/
template<typename datatype>
void KdTree<datatype>::findNeighbors(datatype *v, int n_nn, int *neighdex, double *dd){
  /**
   * @param v the point whose neighbors you want to find
   *
   * @param n_nn the number of nearest neighbors you want to find
   *
   * @param neighdex this is where the indices of the nearest neighbor points will be stored
   *
   * @param dd this is where the distances to the nearest neighbors will be stored
   *
   * neighbors will be returned in ascending order of distance
   *
   * note that distance is defined by the function which was passed into the constructor
  */
  
  int i,start,order[3];
  double dorder[3];
  
  _neighborCandidates=new int[n_nn];
  _neighborDistances=new double[n_nn];
  _neighborsFound=0;
  _neighborsWanted=n_nn;
  
  for(i=0;i<n_nn;i++)_neighborDistances[i]=-1.0;
  
  start=_findNode(v);
  
  _neighborDistances[0]=_distance(data[start],v,_dimensions);
  _neighborCandidates[0]=start;
  _neighborsFound=1;
  
  if(_tree[start][3]>=0)dorder[2]=_distance(data[_tree[start][3]],v,_dimensions);
  else dorder[2]=-1.0;
  order[2]=3;
  
  if(_tree[start][1]>=0)dorder[0]=_distance(data[_tree[start][1]],v,_dimensions);
  else dorder[0]=-1.0;
  order[0]=1;
  
  if(_tree[start][2]>=0)dorder[1]=_distance(data[_tree[start][2]],v,_dimensions);
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
  
  /*if(_tree[start][3]>=0){
    _lookForNeighbors(v,_tree[start][3],start);
  }
  if(_tree[start][1]>=0){
    _lookForNeighbors(v,_tree[start][1],start);
  }
  if(_tree[start][2]>=0){
    _lookForNeighbors(v,_tree[start][2],start);
  }*/
  
  for(i=0;i<n_nn;i++){
    neighdex[i]=_neighborCandidates[i];
    dd[i]=_neighborDistances[i];
  }
  
  delete [] _neighborCandidates;
  delete [] _neighborDistances;
  
}

/**
 * @brief This method actually looks for the neighbors, determining whether or not to descend branches of the tree
*/
template<typename datatype>
void KdTree<datatype>::_lookForNeighbors(datatype *v, int consider, int from){
  /**
   * @param v the point whose neighbors you are looking for
   *
   * @param consider the index of the data point you are considering as a possible nearest neighbor
   *
   * @param from the index of the point you last considered as a nearest neighbor (so the search does not backtrack along the tree)
   *
   * The class KdTree keeps track of how many neighbors you want and how many neighbors you have found and what their
   * distances from v are in the class member variables _neighborsWanted, _neighborsFound, _neighborCandidates,
   * and _neighborDistances
  */
  
  int i,j,going;
  double dd;
  
  dd=_distance(data[consider],v,_dimensions);
  
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

/**
 * @brief return the number of data points stored in the tree
*/
template <typename datatype>
int KdTree<datatype>::getPoints(){
  return _pts;
}

template <typename dtyi, typename dtyo>
GaussianProcess<dtyi,dtyo>::~GaussianProcess(){
  
  if(_useMaxMin==1){    
    if(_calledInterpolate==1)delete [] _vv;
  }
  
    delete _kdTreePtr;
    delete _function;
  
  if(_calledInterpolate==1){
    delete [] _neighbors;
    delete [] _neighborDistances;
    delete [] _covarianceTestPoint;
    _covariance.resize(0,0);
  }
  
}

/**
 * @brief This is the constructor you call if you want the positions of your data points normalized by the span of each dimension
*/

template <typename dtyi, typename dtyo>
GaussianProcess<dtyi,dtyo>::GaussianProcess(int dd, int pp, ndarray::Array<dtyi,2,2> datain,\
ndarray::Array<dtyi,1,1> mn, ndarray::Array<dtyi,1,1> mx, ndarray::Array<dtyo,1,1> ff){
 
 /**
  * @param dd the number of dimensions of your data points
  *
  * @param pp the number of data points you are inputting
  *
  * @param datain an ndarray containing the data points; the ith row of datain is the ith data point
  *
  * @param mn a one-dimensional ndarray containing the minimum values of each dimension (for normalizing the positions of data points)
  *
  * @param mx a one-dimensional ndarray containing the maximum values of each dimension (for normalizing the positions of data points)
  *
  * @param ff a one-dimensional ndarray containing the values of the scalar function associated with each data point.  
  * This is the function you are interpolating
 */
  
  int i,j;
  
  _dimensions=dd;
  _pts=pp;
  _room=_pts;
  _roomStep=5000;
  
  _krigingParameter=dtyo(1.0);
    
  _covariogram=GPfn::expCovariogram;
  _distance=GPfn::euclideanDistance;
  
  _calledInterpolate=0;

 _lambda=dtyo(1.0e-5);
 _krigingParameter=dtyo(1.0);
 
 _max=new dtyi[_dimensions];
 _min=new dtyi[_dimensions];
 for(i=0;i<_dimensions;i++){
   _max[i]=mx(i);
   _min[i]=mn(i);
 }
  
  _useMaxMin=1;
  _data=new dtyi*[_pts];
  for(i=0;i<_pts;i++){
    _data[i]=new dtyi[_dimensions];
    for(j=0;j<_dimensions;j++){
      _data[i][j]=(datain(i,j)-_min[j])/(_max[j]-_min[j]); //note the normalization by _max-_min in each dimension
    }
  }
   
  _kdTreePtr=new KdTree<dtyi>(_dimensions,_pts,_data,_distance);
  
  for(i=0;i<_pts;i++)delete [] _data[i];
  delete [] _data;
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();
  
  _function=new dtyo[_pts];
  for(i=0;i<_pts;i++)_function[i]=ff(i);
  
  _typeOfCovariogram=squaredExp;
  _nHyperParameters=1;
  _hyperParameters=new double[1];
  _hyperParameters[0]=1.0;
  
  interpolationTime=0.0;
  interpolationCount=0;
  neighborSearchTime=0.0;
  inversionTime=0.0;
  iterationTime=0.0;
  varSolveTime=0.0;
  

}

/**
 @brief This is the constructor you call if you do not wish to normalize the positions of your data points
*/

template <typename dtyi, typename dtyo>
GaussianProcess<dtyi,dtyo>::GaussianProcess(int dd, int pp, ndarray::Array<dtyi,2,2> datain, \
ndarray::Array<dtyo,1,1> ff){
 
   /**
  * @param dd the number of dimensions of your data points
  *
  * @param pp the number of data points you are inputting
  *
  * @param datain an ndarray containing the data points; the ith row of datain is the ith data point
  *
  * @param ff a one-dimensional ndarray containing the values of the scalar function associated with each data point.  
  * This is the function you are interpolating
 */
    
  int i,j;

  _dimensions=dd;
  _pts=pp;
  _room=_pts;
  _roomStep=5000;
  
  _function=new dtyo[_pts];
  for(i=0;i<_pts;i++)_function[i]=ff(i);
  _krigingParameter=dtyo(1.0);
  
  _covariogram=GPfn::expCovariogram;
  _distance=GPfn::euclideanDistance;
  
  _calledInterpolate=0;
  
  _lambda=dtyo(1.0e-5);
  
  _useMaxMin=0;
  
  
  _data=new dtyi*[_pts];
  for(i=0;i<_pts;i++){
     _data[i]=new dtyi[_dimensions];
     for(j=0;j<_dimensions;j++){
       _data[i][j]=datain(i,j);
     }
  }
  
  _kdTreePtr=new KdTree<dtyi>(_dimensions,_pts,_data,_distance);
  
  for(i=0;i<_pts;i++){
    delete [] _data[i];
  }
  delete _data;
  
  _data=_kdTreePtr->data;
  _pts=_kdTreePtr->getPoints();
  
  _typeOfCovariogram=squaredExp;
  _nHyperParameters=1;
  _hyperParameters=new double[1];
  _hyperParameters[0]=1.0;
  
  interpolationTime=0.0;
  interpolationCount=0;
  neighborSearchTime=0.0;
  inversionTime=0.0;
  iterationTime=0.0;
  varSolveTime=0.0;
  
}

/**
 * @brief Assign a value to the Kriging paramter
*/

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setKrigingParameter(dtyo kk){
  
  /**
   * @param kk the value assigned to the Kriging parameters
   *
  */
  
  _krigingParameter=kk;
  
  
}

/**
 @brief Interpolate the function value at one point using a specified number of nearest neighbors
*/

template <typename dtyi, typename dtyo>
dtyo GaussianProcess<dtyi,dtyo>::interpolate(ndarray::Array<dtyi,1,1> vin, ndarray::Array<dtyo,1,1> variance, int kk){
  
  /**
   * @param vin a one-dimensional ndarray representing the point at which you want to interpolate the function
   *
   * @param variance a one-dimensional ndarray.  The value of the variance predicted by the Gaussina process will be stored in the zeroth element
   *
   * @param kk the number of nearest neighbors to be used in the interpolation
   *
   * the interpolated value of the function will be returned at the end of this method
  */
  
  int i,j;
  dtyo fbar,mu;
  double before,after,aa,bb;
  
  before=double(::time(NULL));
  
  if(_calledInterpolate==0 || kk!=_numberOfNeighbors){
  //if this is not the first time you have called this method, the code must make sure that the
  //arrays it uses are large enough to accommodate the number of nearest neighbors you asked for
  
     if(_calledInterpolate==1){
       delete [] _covarianceTestPoint;
       delete [] _neighbors;
       delete [] _neighborDistances;
       
     }
     
     _covarianceTestPoint=new dtyo[kk];
     _covariance.resize(kk,kk);
     _bb.resize(kk,1);
     _xx.resize(kk,1);
     
     _neighbors=new int[kk];
     _neighborDistances=new double[kk];
     
     _numberOfNeighbors=kk;
  }
  
  if(_calledInterpolate==0){
    _vv=new dtyi[_dimensions];
  }
  
  if(_useMaxMin==1){
    //if you constructed this Gaussian process with minimum and maximum values for the dimensions of your parameter space,
    //the point you are interpolating must be scaled to match the data so that the selected nearest neighbors are appropriate
    
    for(i=0;i<_dimensions;i++)_vv[i]=(vin(i)-_min[i])/(_max[i]-_min[i]);
  }
  else{
    for(i=0;i<_dimensions;i++){
      _vv[i]=vin(i);
    }
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
  
  /*if(_typeOfCovariogram==neuralNetwork){
    printf("time to write lambda %e\n",_lambda);
    for(i=0;i<_numberOfNeighbors;i++){
      printf("%le\n",_covariance(0,i));
    }
    exit(1);
  }*/
  
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

/**
 * @brief Output the indices of data points curently stored in the _neighbors array
*/
template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::getNeighbors(ndarray::Array<int,1,1> v){
  int i;
  if(_calledInterpolate==0){
    std::cout<<"You cannot call getNeighbors; you have not called interpolate at all\n";
    //printf("You cannot call print_nn; you haven't called interpolate at all\n");
  }
  else{
    for(i=0;i<_numberOfNeighbors;i++)v(i)=_neighbors[i];
  }
}

/**
 * @brief set the value of the hyperparameter _lambda
 *
 * _lambda is a parameter meant to represent the characteristic variance
 * of the function you are interpolating.  Currently, it is a scalar such that
 * all data points must have the same characteristic variance.  Future iterations
 * of the code may want to promote _lambda to an array so that different data points
 * can have different variances.
*/
template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setLambda(dtyo ll){

  _lambda=ll;

}

/**
 * @brief Output a specified row of the last computed covariance matrix
*/

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::getCovarianceRow(int dex, ndarray::Array<dtyo,1,1> v){
  /**
   * @param dex the row that you want
   *
   * @param v a one-dimensiona ndarray where the row will be stored
  */

  int i;
  if(_calledInterpolate==0){
    std::cout<<"You cannot call getCovarianceRow; you have not called interpolate\n";
    //printf("You can't call print gg_row; you haven't called interpolate\n");
  }
  else{
    for(i=0;i<_numberOfNeighbors;i++)v(i)=_covariance(dex,i);
  }
}

/**
 * @brief Add a point to the pool of data used by GaussianProcess for interpolation
*/
template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::addPoint(ndarray::Array<dtyi,1,1> vin, dtyo f){

 /**
  * @param vin a one-dimensional ndarray storing the point in parameter space that you are adding
  *
  * @param f the value of the function at that point
 */

  int i;
  dtyi *v;
  dtyo *buff;
  
  v=new dtyi[_dimensions];
  for(i=0;i<_dimensions;i++){
    v[i]=vin(i);
    if(_useMaxMin==1){
      v[i]=(v[i]-_min[i])/(_max[i]-_min[i]);
    }
    
  }
  
  if(_pts==_room){
    buff=new dtyo[_room];
    for(i=0;i<_room;i++){
      buff[i]=_function[i];
    }
    delete [] _function;
    
    _room+=_roomStep;
    _function=new dtyo[_room];
    for(i=0;i<_pts;i++){
      _function[i]=buff[i];
    }
    delete [] buff;
  }
  _function[_pts]=f;
  
  _kdTreePtr->addPoint(v);
  _pts=_kdTreePtr->getPoints();
  _data=_kdTreePtr->data;
  
  delete [] v;
}

/**
 * @brief Run KdTree::testTree to make sure that the KD Tree is properly constructed.  Returns 1 if it is.
*/

template <typename dtyi, typename dtyo>
int GaussianProcess<dtyi,dtyo>::testKdTree(){

  return _kdTreePtr->testTree();

}

/**
 * @brief Set the values of the hyperparameters governing the covariogram.  The method knows how many there should be.
*/

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setHyperParameters(ndarray::Array<double,1,1> hyin){
  
  /**
   * @param hyin a one-dimensional ndarray containing the hyperparameter values to be set.
   * 
   * The number of parameters in hyin should correspond to the number of parameters associated with the chosen type
   * of covariogram
  */
  
  int i;
  for(i=0;i<_nHyperParameters;i++){
    _hyperParameters[i]=hyin(i);
  }
  
  
}

/**
 * @brief Print the the time spent on neighbor searches, interpolation, matrix inversion, iterating over matrix indices, and finding variances
*/
template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::getTimes(){
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

/**
 * @brief Reset the times being tracked inside interpolate
*/
template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::resetTimes(){
  interpolationTime=0.0;
  neighborSearchTime=0.0;
  inversionTime=0.0;
  varSolveTime=0.0;
  iterationTime=0.0;
  interpolationCount=0;
}

/**
* @brief Select the type of covariogram from those enumerated in GaussianProcess.h
*/
template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setCovariogramType(int ii){
  
  /**
   * @param ii The type of covariogram you want to use
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
  
  int i;
  
  delete [] _hyperParameters;
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
  
  _hyperParameters=new double[_nHyperParameters];
  for(i=0;i<_nHyperParameters;i++){
    _hyperParameters[i]=1.0;
  }
  
  _typeOfCovariogram=ii;
  
}

/**
 * @brief Interpolate a list of query points using all of the input data (rather than nearest neighbors)
*/
template<typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::batchInterpolate(ndarray::Array<dtyi,2,2> queries, ndarray::Array<dtyo,1,1> mu, \
ndarray:: Array<dtyo,1,1> variance, int nQueries){
  
  /**
   * @param queries a 2-dimensional ndarray containing the points to be interpolated.  queries[i][j] is the jth component of the ith point
   *
   * @param mu a 1-dimensional ndarray where the interpolated function values will be stored
   *
   * @param variance a 2-dimensional ndarray where the corresponding variances in the function value will be stored
   *
   * @param nQueries the number of points being interpolated
   *
   * This method will attempt to construct a _pts X _pts covariance matrix C and solve the problem Cx=b.
   * Be wary of using it in the case where _pts is very large.
   *
   * This version of the method will also return variances for all of the query points.  That is a very time consuming
   * calculation relative to just returning estimates for the function.  Consider calling the version of this method
   * that does not calculate variances (below).  The difference in speed is an order of magnitude in the case of
   * 189 data points and 1 million queries.
   *
  */
  
  int i,j,ii;
  double aa,bb;
  dtyi *v1;
  dtyo fbar;
  Eigen::Matrix <dtyo,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <dtyi,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  
  interpolationTime=0.0;
  varSolveTime=0.0;
  
  bb=double(::time(NULL));
  v1=new dtyi[_dimensions];
  
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
  fbar=fbar/dtyo(_pts);
  
  //std::cout<<"fbar "<<fbar<<"\n";
  
  for(i=0;i<_pts;i++){
    batchbb(i,0)=_function[i]-fbar;
  }
  batchxx=_llt.solve(batchbb);
  
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
  aa=double(::time(NULL));
  interpolationTime+=aa-bb;
  
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
  
  delete [] v1;

}

/**
 * @brief Interpolate a list of points using all of the data. Do not return variances for the interpolation.
*/
template<typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::batchInterpolate(ndarray::Array<dtyi,2,2> queries, ndarray::Array<dtyo,1,1> mu,\
 int nQueries){

  /**
   * @param queries a 2-dimensional ndarray containing the points to be interpolated.  queries[i][j] is the jth component of the ith point
   *
   * @param mu a 1-dimensional ndarray where the interpolated function values will be stored
   *
   * @param nQueries the number of points being interpolated
   *
   * This method will attempt to construct a _pts X _pts covariance matrix C and solve the problem Cx=b.
   * Be wary of using it in the case where _pts is very large.
   *
   * This version of the method does not return variances.  It is an order of magnitude faster than the version of the method
   * that does return variances (timing done on a case with 189 data points and 1 million query points).
   *
  */

  int i,j,ii;
  double aa,bb;
  dtyi *v1;
  dtyo fbar;
  Eigen::Matrix <dtyo,Eigen::Dynamic,Eigen::Dynamic> batchCovariance,batchbb,batchxx;
  Eigen::Matrix <dtyi,Eigen::Dynamic,Eigen::Dynamic> queryCovariance;
  
  interpolationTime=0.0;
  varSolveTime=0.0;
  
  bb=double(::time(NULL));
  v1=new dtyi[_dimensions];
  
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
  fbar=fbar/dtyo(_pts);
  
  //std::cout<<"fbar "<<fbar<<"\n";
  
  for(i=0;i<_pts;i++){
    batchbb(i,0)=_function[i]-fbar;
  }
  batchxx=_llt.solve(batchbb);
  
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
  aa=double(::time(NULL));
  interpolationTime+=aa-bb;
  
  //std::cout<<"done with interpolation\n";
 
  
  delete [] v1;

}
/**
* @brief This method will interpolate the function on a data point for purposes of optimizing hyper parameters
*/
template <typename dtyi, typename dtyo>
dtyo GaussianProcess<dtyi,dtyo>::selfInterpolate(int dex, ndarray::Array<dtyo,1,1> variance, int kk){
  
  /**
   * @param dex the index of the point you wish to self interpolate
   *
   * @param variance a one-dimensional ndarray.  The value of the variance predicted by the Gaussina process will be stored in the zeroth element
   *
   * @param kk the number of nearest neighbors to be used in the interpolation
   *
   * The interpolated value of the function will be returned at the end of this method
   *
   * This method ignores the point on which you are interpolating when requesting nearest neighbors
   *
  */
  
  int i,j;
  dtyo fbar,mu;
  double before,after,aa,bb,*selfDistances;
  int *selfNeighbors;
  
  before=double(::time(NULL));
  
  if(_calledInterpolate==0 || kk!=_numberOfNeighbors){
  //if this is not the first time you have called this method, the code must make sure that the
  //arrays it uses are large enough to accommodate the number of nearest neighbors you asked for
  
     if(_calledInterpolate==1){
       delete [] _covarianceTestPoint;
       delete [] _neighbors;
       delete [] _neighborDistances;
       
     }
     
     _covarianceTestPoint=new dtyo[kk];
     _covariance.resize(kk,kk);
     _bb.resize(kk,1);
     _xx.resize(kk,1);
     
     _neighbors=new int[kk];
     _neighborDistances=new double[kk];
     
     _numberOfNeighbors=kk;
  }
  
  selfNeighbors=new int[_numberOfNeighbors+1];
  selfDistances=new double[_numberOfNeighbors+1];
  
  if(_calledInterpolate==0){
    _vv=new dtyi[_dimensions];
  }
  
  //we don't use _useMaxMin because _data has already been normalized
    for(i=0;i<_dimensions;i++){
      _vv[i]=_data[dex][i];
    }
  
  
  bb=double(::time(NULL));
  _kdTreePtr->findNeighbors(_vv,_numberOfNeighbors+1,selfNeighbors,selfDistances);
  aa=double(::time(NULL));
  
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
  
  /*if(_typeOfCovariogram==neuralNetwork){
    printf("time to write lambda %e\n",_lambda);
    for(i=0;i<_numberOfNeighbors;i++){
      printf("%le\n",_covariance(0,i));
    }
    exit(1);
  }*/
  
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
  
  delete [] selfNeighbors;
  delete [] selfDistances;
  
  return mu;
}


}}}

#define INSTANTIATEGP(dtyi,dtyo) \
	template class lsst::afw::math::GaussianProcess<dtyi,dtyo>;

INSTANTIATEGP(double,double);


