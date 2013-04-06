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
 
#include <iostream>
#include <cmath>
#include "lsst/afw/math/detail/GaussianProcessFunctions.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gaussianProcess {



template<typename T>
double euclideanDistance(ndarray::Array<const T,1,1> const &v1, 
                         ndarray::Array<const T,1,1> const &v2, 
			 int d_dim)
{

  int i;
  double dd;
  dd=0.0;
  for(i=0;i<d_dim;i++){
    dd+=double(v1[i]-v2[i])*double(v1[i]-v2[i]);
  }
  
  return ::sqrt(dd);
}

template <typename T>
void mergeSort(ndarray::Array<T,1,1> const &insort, ndarray::Array<int,1,1> const &indices, int el){

  
  int i,k;
  T nn;
  
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
    i=mergeScanner<T>(insort,indices,el/2,el);
  
    if(i>1){
      mergeSort<T>(insort,indices,i);
    }
    
    if(i<el-2){
      //mergeSort<T>(&insort[i+1],&indices[i+1],el-i-1);
      mergeSort<T>(insort[ndarray::view(i+1,insort.template getSize<0>())],\
      indices[ndarray::view(i+1,indices.template getSize<0>())],el-i-1);
    }
  
  }

}

template <typename T>
int mergeScanner(ndarray::Array<T,1,1> const &m, ndarray::Array<int,1,1> const &indices, int dex, int el){

  int i,j,k,newdex;
  T nn;
    
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

}}}}}

#define gpn lsst::afw::math::detail::gaussianProcess

#define INSTANTIATEDETAIL(T) \
        template void \
        gpn::mergeSort<T>(ndarray::Array<T,1,1> const &,\
                          ndarray::Array<int,1,1> const &, int el); \
        template T gpn::euclideanDistance<T>(ndarray::Array<const T,1,1> const &,\
                                             ndarray::Array<const T,1,1> const &, int);

INSTANTIATEDETAIL(double);

