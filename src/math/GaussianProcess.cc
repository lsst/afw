//  - * -  LSST - C++   - * -

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
 * see  < http://www.lsstcorp.org/LegalNotices/ > .
 */

#include  <iostream>
#include  <cmath>

#include "lsst/afw/math/GaussianProcess.h"

using namespace std;


namespace lsst {
namespace afw {
namespace math {

GaussianProcessTimer::GaussianProcessTimer()
{
    _interpolationCount = 0;
    _iterationTime = 0.0;
    _eigenTime = 0.0;
    _searchTime = 0.0;
    _varianceTime = 0.0;
    _totalTime = 0.0;

}

void GaussianProcessTimer::reset()
{
    _interpolationCount = 0;
    _iterationTime = 0.0;
    _eigenTime = 0.0;
    _searchTime = 0.0;
    _varianceTime = 0.0;
    _totalTime = 0.0;

}

void GaussianProcessTimer::start()
{
    _before = double(lsst::daf::base::DateTime::now().get()*24*60*60);
    _beginning = _before;
}

void GaussianProcessTimer::addToEigen()
{
    double after;
    after = double(lsst::daf::base::DateTime::now().get()*24*60*60);
    _eigenTime += after - _before;
    _before = after;
}

void GaussianProcessTimer::addToVariance()
{
    double after;
    after = double(lsst::daf::base::DateTime::now().get()*24*60*60);
    _varianceTime += after - _before;
    _before = after;
}

void GaussianProcessTimer::addToSearch()
{
    double after;
    after = double(lsst::daf::base::DateTime::now().get()*24*60*60);
    _searchTime += after - _before;
    _before = after;
}

void GaussianProcessTimer::addToIteration()
{
    double after;
    after = double(lsst::daf::base::DateTime::now().get()*24*60*60);
    _iterationTime += after - _before;
    _before = after;
}

void GaussianProcessTimer::addToTotal(int i)
{
    double after;
    after = double(lsst::daf::base::DateTime::now().get()*24*60*60);
    _totalTime += after - _beginning;
    _interpolationCount += i;
}

void GaussianProcessTimer::display(){
    std::cout << "\nSearch time " << _searchTime << "\n";
    std::cout << "Eigen time " << _eigenTime << "\n";
    std::cout << "Variance time " << _varianceTime << "\n";
    std::cout << "Iteration time " << _iterationTime << "\n";
    std::cout << "Total time " << _totalTime << "\n";
    std::cout << "Number of interpolations " << _interpolationCount << "\n";
}

template  < typename T >
void KdTree < T > ::Initialize(ndarray::Array < T,2,2 >  const &dt)
{
    int i;

    _pts = dt.template getSize < 0 > ();
    _dimensions = dt.template getSize < 1 > ();

    //a buffer to use when first building the tree
    _inn = allocate(ndarray::makeVector(_pts));

    _roomStep = 5000;
    _room = _pts;

    _data = allocate(ndarray::makeVector(_room,_dimensions));

    _data.deep() = dt;

    _tree = allocate(ndarray::makeVector(_room,4));

    for (i = 0 ; i < _pts ; i++ ) {
      _inn[i] = i;
    }

    _organize(_inn,_pts, -1, -1);


    i = _testTree();
    if (i == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Failed to properly initialize KdTree\n");
    }

}

template < typename T >
void KdTree < T > ::findNeighbors(ndarray::Array < int,1,1 >  neighdex,
                                  ndarray::Array < double,1,1 >  dd,
                                  ndarray::Array < const T,1,1 >  const &v,
                                  int n_nn) const
{

    if(n_nn > _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for more neighbors than kd tree contains\n");
    }

    if(n_nn <= 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for zero or a negative number of neighbors\n");
    }

    if(neighdex.getNumElements() != n_nn){
       throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                         "Size of neighdex does not equal n_nn in KdTree.findNeighbors\n");
    }

    if(dd.getNumElements() != n_nn){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Size of dd does not equal n_nn in KdTree.findNeighbors\n");
    }

    int i,start;

    _neighborCandidates = allocate(ndarray::makeVector(n_nn));
    _neighborDistances = allocate(ndarray::makeVector(n_nn));
    _neighborsFound = 0;
    _neighborsWanted = n_nn;

    for (i = 0; i < n_nn; i++ ) _neighborDistances[i] =  -1.0;

    start = _findNode(v);

    _neighborDistances[0] = _distance(v,_data[start]);
    _neighborCandidates[0] = start;
    _neighborsFound = 1;


    for (i = 1; i < 4; i++ ) {
        if(_tree[start][i] >= 0){
            _lookForNeighbors(v,_tree[start][i], start);
        }
    }

    for (i = 0 ; i < n_nn ; i++ ) {
        neighdex[i] = _neighborCandidates[i];
        dd[i] = _neighborDistances[i];
    }
}

template  < typename T >
T KdTree < T > ::getData(int ipt, int idim) const
{
    if(ipt >= _pts || ipt < 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for point that does not exist in KdTree\n");
    }

    if(idim >= _dimensions || idim < 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "KdTree does not have points of that many dimensions\n");
    }

    return _data[ipt][idim];
}

template  < typename T >
ndarray::Array < T,1,1 >  KdTree < T > ::getData(int ipt) const
{
    if(ipt >= _pts || ipt<0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for point that does not exist in KdTree\n");
    }

    return _data[ipt];
}

template  < typename T >
void KdTree < T > ::addPoint(ndarray::Array < const T,1,1 >  const &v)
{

    if(v.getNumElements() != _dimensions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You are trying to add a point of the incorrect dimensionality to KdTree\n");
    }

    int i,j,node,dim,dir;

    node = _findNode(v);
    dim = _tree[node][DIMENSION] + 1;
    if(dim == _dimensions)dim = 0;

    if(_pts == _room){

        ndarray::Array < T,2,2 >  dbuff = allocate(ndarray::makeVector(_pts, _dimensions));
        ndarray::Array < int,2,2 >  tbuff = allocate(ndarray::makeVector(_pts, 4));

        dbuff.deep() = _data;
        tbuff.deep() = _tree;

        _room += _roomStep;

        _tree = allocate(ndarray::makeVector(_room, 4));
        _data = allocate(ndarray::makeVector(_room, _dimensions));


        for (i = 0; i < _pts; i++ ) {
            for (j = 0; j < _dimensions; j++ ) _data[i][j] = dbuff[i][j];
            for (j = 0; j < 4; j++ ) _tree[i][j] = tbuff[i][j];
        }
    }

    _tree[_pts][DIMENSION] = dim;
    _tree[_pts][PARENT] = node;
    i = _tree[node][DIMENSION];

    if(_data[node][i] > v[i]){
        if(_tree[node][LT] >= 0){
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
            "Trying to add to KdTree in a node that is already occupied\n");
        }
        _tree[node][LT] = _pts;
        dir=LT;
    }
    else{
        if(_tree[node][GEQ] >= 0){
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
            "Trying to add to KdTree in a node that is already occupied\n");
        }
        _tree[node][GEQ] = _pts;
        dir=GEQ;
    }
    _tree[_pts][LT] =  -1;
    _tree[_pts][GEQ] =  -1;
    for (i = 0; i < _dimensions; i++ ) {
        _data[_pts][i] = v[i];
    }

    _pts++;


    i = _walkUpTree(_tree[_pts-1][PARENT], dir, _pts-1);
    if (i != _masterParent){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
        "Adding to KdTree failed\n");
    }

}

template  < typename T >
int KdTree < T > ::getPoints() const
{
    return _pts;
}

template  < typename T >
void KdTree < T > ::getTreeNode(ndarray::Array < int,1,1 >  const &v,
                                int dex) const
{
    if(dex < 0 || dex >= _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for tree information on point that does not exist\n");
    }

    if(v.getNumElements() != 4){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Need to pass a 4-element ndarray into KdTree.getTreeNode()\n");
    }

    v[0] = _tree[dex][DIMENSION];
    v[1] = _tree[dex][LT];
    v[2] = _tree[dex][GEQ];
    v[3] = _tree[dex][PARENT];
}

template  < typename T >
int KdTree < T > ::_testTree() const{

    if(_pts == 1 && _tree[0][PARENT] < 0 && _masterParent==0){
        // there is only one point in the tree
        return 1;
    }

    int i,j,output;
    std::vector < int >  isparent;

    output=1;

    for (i = 0; i < _pts; i++ ) isparent.push_back(0);

    j = 0;
    for (i = 0; i < _pts; i++ ) {
        if(_tree[i][PARENT] < 0)j++;
    }
    if(j != 1){
        std::cout << "_tree FAILURE " << j << " _masterParents\n";
        return 0;
    }

    for (i = 0; i < _pts; i++ ) {
        if(_tree[i][PARENT] >= 0){
            isparent[_tree[i][PARENT]]++;
        }
    }

    for (i = 0; i < _pts; i++ ) {
        if(isparent[i] > 2){
            std::cout << "_tree FAILURE " << i << " is parent to " << isparent[i] << "\n";
            return 0;
        }
    }


    for (i = 0; i < _pts; i++ ) {
        if(_tree[i][PARENT] >= 0) {
            if(_tree[_tree[i][PARENT]][LT] == i)j = LT;
            else j = GEQ;
            output = _walkUpTree(_tree[i][PARENT], j, i);

            if(output != _masterParent){
                return 0;
            }
        }
    }

    if(output != _masterParent) return 0;
    else return 1;
}

template  < typename T >
void KdTree < T > ::_organize(ndarray::Array < int,1,1 >  const &use,
                              int ct,
                              int parent,
                              int dir
                              )
{

    int i,j,k,l,idim,daughter;
    T mean,var,varbest;


    std::vector <  std::vector < T >   >  toSort;
    std::vector < T >  toSortElement;

    if(ct > 1){
      //below is code to choose the dimension on which the available points
      //have the greates variance.  This will be the dimension on which
      //the daughter node splits the data
        idim=0;
	varbest=-1.0;
        for (i = 0; i < _dimensions; i++ ) {
            mean = 0.0;
            var = 0.0;
            for (j = 0; j < ct; j++ ) {
                mean += _data[use[j]][i];
                var += _data[use[j]][i]*_data[use[j]][i];
            }
            mean = mean/double(ct);
            var = var/double(ct) - mean*mean;
            if(i == 0 || var > varbest ||
	       (var == varbest && parent >= 0 && i > _tree[parent][DIMENSION])) {
                    idim = i;
                    varbest = var;
            }
        }//for(i = 0;i < _dimensions;i++ )

        //we need to sort the available data points according to their idim - th element
        //but we need to keep track of their original indices, so we can refer back to
        //their positions in _data.  Therefore, the code constructs a 2 - dimensional
        //std::vector.  The first dimension contains the element to be sorted.
        //The second dimension contains the original index information.
        //After sorting, the (rearranged) index information will be stored back in the
        //ndarray use[]

        toSortElement.push_back(_data[use[0]][idim]);
        toSortElement.push_back(T(use[0]));
        toSort.push_back(toSortElement);
        for(i = 1; i < ct; i++ ) {
            toSortElement[0] = _data[use[i]][idim];
            toSortElement[1] = T(use[i]);
            toSort.push_back(toSortElement);
        }

        std::stable_sort(toSort.begin(), toSort.end());

        k = ct/2;
        l = ct/2;
        while(k > 0 && toSort[k][0] == toSort[k - 1][0]) k--;

        while(l < ct - 1 && toSort[l][0] == toSort[ct/2][0]) l++;

        if((ct/2 - k) < (l - ct/2) || l == ct - 1) j = k;
        else j = l;

        for(i = 0; i < ct; i++ ) {
            use[i] = int(toSort[i][1]);
        }
        daughter = use[j];

        if(parent >= 0)_tree[parent][dir] = daughter;
        _tree[daughter][DIMENSION] = idim;
        _tree[daughter][PARENT] = parent;

        if(j < ct - 1){
            _organize(use[ndarray::view(j + 1,use.getSize < 0 > ())], ct - j - 1, daughter, GEQ);
        }
        else _tree[daughter][GEQ] =  -1;

        if(j > 0){
            _organize(use, j, daughter, LT);
        }
        else _tree[daughter][LT] =  -1;

    }//if(ct > 1)
    else{
        daughter = use[0];

        if(parent >= 0) _tree[parent][dir] = daughter;

        idim = _tree[parent][DIMENSION] + 1;

        if(idim >= _dimensions)idim = 0;

        _tree[daughter][DIMENSION] = idim;
        _tree[daughter][LT] =  - 1;
        _tree[daughter][GEQ] =  - 1;
        _tree[daughter][PARENT] = parent;

    }

    if(parent ==  - 1){
        _masterParent = daughter;
    }

}

template  < typename T >
int KdTree < T > ::_findNode(ndarray::Array < const T,1,1 >  const &v) const
{
    int consider,next,dim;

    dim = _tree[_masterParent][DIMENSION];

    if(v[dim] < _data[_masterParent][dim]) consider = _tree[_masterParent][LT];
    else consider = _tree[_masterParent][GEQ];

    next = consider;

    while(next >= 0){

        consider = next;

        dim = _tree[consider][DIMENSION];
        if(v[dim] < _data[consider][dim]) next = _tree[consider][LT];
        else next = _tree[consider][GEQ];

    }

    return consider;
}

template < typename T >
void KdTree < T > ::_lookForNeighbors(ndarray::Array < const T,1,1 >  const &v,
                                      int consider,
                                      int from) const
{

    int i,j,going;
    double dd;

    dd = _distance(v, _data[consider]);

    if(_neighborsFound < _neighborsWanted || dd < _neighborDistances[_neighborsWanted -1]) {

        for(j = 0; j < _neighborsFound && _neighborDistances[j] < dd; j++ );

        for(i = _neighborsWanted - 1; i > j; i-- ){
            _neighborDistances[i] = _neighborDistances[i - 1];
            _neighborCandidates[i] = _neighborCandidates[i - 1];
        }

        _neighborDistances[j] = dd;
        _neighborCandidates[j] = consider;

        if(_neighborsFound < _neighborsWanted) _neighborsFound++;
    }

    if(_tree[consider][PARENT] == from){
    //you came here from the parent

        i = _tree[consider][DIMENSION];
        dd = v[i] - _data[consider][i];
        if((dd <= _neighborDistances[_neighborsFound - 1] || _neighborsFound < _neighborsWanted)
            && _tree[consider][LT] >= 0){

              _lookForNeighbors(v, _tree[consider][LT], consider);
        }

        dd = _data[consider][i] - v[i];
        if((dd <= _neighborDistances[_neighborsFound - 1] || _neighborsFound < _neighborsWanted)
            && _tree[consider][GEQ] >= 0){

            _lookForNeighbors(v, _tree[consider][GEQ], consider);
        }
    }
    else{
    //you came here from one of the branches

        //descend the other branch
        if(_tree[consider][LT] == from){
            going = GEQ;
        }
        else{
            going = LT;
        }

        j = _tree[consider][going];

        if(j >= 0){
            i = _tree[consider][DIMENSION];

            if(going == 1) dd = v[i] - _data[consider][i];
            else dd = _data[consider][i] - v[i];

            if(dd <= _neighborDistances[_neighborsFound - 1] || _neighborsFound < _neighborsWanted) {
                  _lookForNeighbors(v, j, consider);
            }
        }

        //ascend to the parent
        if(_tree[consider][PARENT] >= 0) {
              _lookForNeighbors(v, _tree[consider][PARENT], consider);
        }
    }
}

template  < typename T >
int KdTree < T > ::_walkUpTree(int target,
                               int dir,
                               int root) const
{
    //target is the node that you are examining now
    //dir is where you came from
    //root is the ultimate point from which you started

    int i,output;

    output = 1;

    if(dir == LT){
        if(_data[root][_tree[target][DIMENSION]] >= _data[target][_tree[target][DIMENSION]]){
            return 0;
        }
    }
    else{
        if(_data[root][_tree[target][DIMENSION]] < _data[target][_tree[target][DIMENSION]]) {
            return 0;
        }
    }

    if(_tree[target][PARENT] >= 0){
        if(_tree[_tree[target][PARENT]][LT] == target)i = LT;
        else i = GEQ;
        output = output*_walkUpTree(_tree[target][PARENT],i,root);

    }
    else{
        output = output*target;
        //so that it will return _masterParent
        //make sure everything is connected to _masterParent
    }
    return output;

}

template  < typename T >
void KdTree < T > ::removePoint(int target)
{

    if(target < 0 || target >= _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You are trying to remove a point that doesn't exist from KdTree\n");
    }

    if(_pts==1){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "There is only one point left in this KdTree.  You cannot call removePoint\n");
    }

    int nl,nr,i,j,k,side;
    int root;

    nl = 0;
    nr = 0;
    if(_tree[target][LT] >= 0){
        nl++ ;
        _count(_tree[target][LT], &nl);
    }

    if(_tree[target][GEQ] >= 0){
        nr++;
        _count(_tree[target][GEQ], &nr);
    }

    if(nl == 0 && nr == 0){

        k = _tree[target][PARENT];

          if(_tree[k][LT] == target) _tree[k][LT] =  - 1;
          else if(_tree[k][GEQ] == target) _tree[k][GEQ] =  - 1;

    }//if target is terminal
    else if((nl == 0 && nr > 0) || (nr == 0 && nl > 0)){

         if(nl == 0) side = GEQ;
         else side = LT;

         k = _tree[target][PARENT];
         if(k >= 0){

             if(_tree[k][LT] == target){
                 _tree[k][LT] = _tree[target][side];
                 _tree[_tree[k][LT]][PARENT] = k;

         }
             else{
                 _tree[k][GEQ] = _tree[target][side];
                 _tree[_tree[k][GEQ]][PARENT] = k;
             }
        }
        else{
            _masterParent = _tree[target][side];
            _tree[_tree[target][side]][PARENT] =  - 1;
        }
    }//if only one side is populated
    else{

        if(nl > nr)side = LT;
        else side = GEQ;

         k = _tree[target][PARENT];
         if(k < 0){
             _masterParent = _tree[target][side];
             _tree[_masterParent][PARENT] =  - 1;
          }
          else{
              if(_tree[k][LT] == target){
                  _tree[k][LT] = _tree[target][side];
                  _tree[_tree[k][LT]][PARENT] = k;
              }
              else{
                   _tree[k][GEQ] = _tree[target][side];
                   _tree[_tree[k][GEQ]][PARENT] = k;
              }
          }

         root = _tree[target][3 - side];

         _descend(root);
    }//if both sides are populated

    if(target < _pts - 1){
         for(i = target + 1;i < _pts;i++ ){
             for(j = 0; j < 4; j++ ) _tree[i - 1][j] = _tree[i][j];

             for(j = 0; j < _dimensions; j++ ) _data[i - 1][j] = _data[i][j];
        }

        for(i = 0; i < _pts; i++ ){
            for(j = 1; j < 4; j++ ){
                if(_tree[i][j] > target) _tree[i][j]--;
             }
        }

        if(_masterParent > target)_masterParent-- ;
    }
    _pts--;

    i = _testTree();
    if (i == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
        "Subtracting from KdTree failed\n");
    }


}

template  < typename T >
void KdTree < T > ::_count(int where, int *ct) const
{
    //a way to count the number of vital elements on a given branch

    if(_tree[where][LT] >= 0) {
        ct[0]++;
        _count(_tree[where][LT], ct);
    }
    if(_tree[where][GEQ] >= 0) {
        ct[0]++;
        _count(_tree[where][GEQ], ct);
    }
}

template  < typename T >
void KdTree < T > ::_reassign(int target)
{

    int where,dir,k;

    where = _masterParent;
    if(_data[target][_tree[where][DIMENSION]] < _data[where][_tree[where][DIMENSION]]) dir = LT;
    else dir = GEQ;

    k = _tree[where][dir];
    while(k >= 0) {
        where = k;
        if(_data[target][_tree[where][DIMENSION]] < _data[where][_tree[where][DIMENSION]]) dir = LT;
        else dir = GEQ;
        k = _tree[where][dir];
    }

    _tree[where][dir] = target;
    _tree[target][PARENT] = where;
    _tree[target][LT] =  - 1;
    _tree[target][GEQ] =  - 1;
    _tree[target][DIMENSION] = _tree[where][DIMENSION] + 1;

    if(_tree[target][DIMENSION] == _dimensions) _tree[target][DIMENSION] = 0;
}

template  < typename T >
void KdTree < T > ::_descend(int root)
{

    if(_tree[root][LT] >= 0) _descend(_tree[root][LT]);
    if(_tree[root][GEQ] >= 0) _descend(_tree[root][GEQ]);

    _reassign(root);
}

template  < typename T >
double KdTree < T > ::_distance(ndarray::Array < const T,1,1 >  const &p1,
                                ndarray::Array < const T,1,1 >  const &p2) const
{

    int i,dd;
    double ans;
    ans = 0.0;
    dd = p1.template getSize< 0 >();

    for(i = 0;i < dd;i++ ) ans += (p1[i] - p2[i])*(p1[i] - p2[i]);

    return ::sqrt(ans);

}


template  < typename T >
GaussianProcess < T > ::GaussianProcess(ndarray::Array < T,2,2 >  const &dataIn,
                                        ndarray::Array < T,1,1 >  const &ff,
                                        std::shared_ptr <  Covariogram < T >   >  const &covarIn)

{
    int i;

    _covariogram = covarIn;

    _pts = dataIn.template getSize < 0 > ();
    _dimensions = dataIn.template getSize < 1 > ();

    _room = _pts;
    _roomStep = 5000;

    _nFunctions = 1;
    _function = allocate(ndarray::makeVector(_pts,1));

    if(ff.getNumElements() != _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You did not pass in the same number of data points as function values\n");
    }

    for(i = 0; i < _pts; i++ ) _function[i][0] = ff[i];

    _krigingParameter = T(1.0);
    _lambda = T(1.0e-5);

    _useMaxMin = 0;


    _kdTree.Initialize(dataIn);

    _pts = _kdTree.getPoints();

}

template  < typename T >
GaussianProcess < T > ::GaussianProcess(ndarray::Array < T,2,2 >  const &dataIn,
                                        ndarray::Array < T,1,1 >  const &mn,
                                        ndarray::Array < T,1,1 >  const &mx,
                                        ndarray::Array < T,1,1 >  const &ff,
                                        std::shared_ptr <  Covariogram < T >   >  const &covarIn
                                        )
{

    int i,j;
    ndarray::Array < T,2,2 >  normalizedData;

    _covariogram = covarIn;

    _pts = dataIn.template getSize < 0 > ();
    _dimensions = dataIn.template getSize < 1 > ();
    _room = _pts;
    _roomStep = 5000;

    if(ff.getNumElements() != _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You did not pass in the same number of data points as function values\n");
    }

    if(mn.getNumElements() != _dimensions || mx.getNumElements() != _dimensions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Your min/max values have different dimensionality than your data points\n");
    }

    _krigingParameter = T(1.0);

    _lambda = T(1.0e-5);
    _krigingParameter = T(1.0);

    _max = allocate(ndarray::makeVector(_dimensions));
    _min = allocate(ndarray::makeVector(_dimensions));
    _max.deep() = mx;
    _min.deep() = mn;
    _useMaxMin = 1;
    normalizedData = allocate(ndarray::makeVector(_pts, _dimensions));
    for(i = 0; i < _pts; i++ ) {
        for(j = 0; j < _dimensions; j++ ) {
            normalizedData[i][j] = (dataIn[i][j] - _min[j])/(_max[j] - _min[j]);
            //note the normalization by _max - _min in each dimension
        }
    }

    _kdTree.Initialize(normalizedData);

    _pts = _kdTree.getPoints();
    _nFunctions = 1;
    _function = allocate(ndarray::makeVector(_pts, 1));
    for(i = 0; i < _pts; i++ )_function[i][0] = ff[i];
}

template  < typename T >
GaussianProcess < T > ::GaussianProcess(ndarray::Array < T,2,2 >  const &dataIn,
                                        ndarray::Array < T,2,2 >  const &ff,
                                        std::shared_ptr <Covariogram < T > >  const &covarIn)
{

    _covariogram = covarIn;

    _pts = dataIn.template getSize < 0 > ();
    _dimensions = dataIn.template getSize < 1 > ();

    _room = _pts;
    _roomStep = 5000;

    if(ff.template getSize < 0 > () != _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You did not pass in the same number of data points as function values\n");
    }

    _nFunctions = ff.template getSize < 1 > ();
    _function = allocate(ndarray::makeVector(_pts, _nFunctions));
    _function.deep() = ff;

    _krigingParameter = T(1.0);

    _lambda = T(1.0e-5);

    _useMaxMin = 0;

    _kdTree.Initialize(dataIn);

    _pts = _kdTree.getPoints();
}

template  < typename T >
GaussianProcess < T > ::GaussianProcess(ndarray::Array < T,2,2 >  const &dataIn,
                                    ndarray::Array < T,1,1 >  const &mn,
                                    ndarray::Array < T,1,1 >  const &mx,
                                    ndarray::Array < T,2,2 >  const &ff,
                                    std::shared_ptr <Covariogram < T > >  const &covarIn
                                    )
{

    int i,j;
    ndarray::Array < T,2,2 >  normalizedData;

    _covariogram = covarIn;

    _pts = dataIn.template getSize < 0 > ();
    _dimensions = dataIn.template getSize < 1 > ();

    _room = _pts;
    _roomStep = 5000;

    if(ff.template getSize < 0 > () != _pts){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You did not pass in the same number of data points as function values\n");
    }

    if(mn.getNumElements() != _dimensions || mx.getNumElements() != _dimensions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Your min/max values have different dimensionality than your data points\n");
    }

    _krigingParameter = T(1.0);

    _lambda = T(1.0e-5);
    _krigingParameter = T(1.0);

    _max = allocate(ndarray::makeVector(_dimensions));
    _min = allocate(ndarray::makeVector(_dimensions));
    _max.deep() = mx;
    _min.deep() = mn;
    _useMaxMin = 1;
    normalizedData = allocate(ndarray::makeVector(_pts,_dimensions));
    for(i = 0; i < _pts; i++ ) {
        for(j = 0; j < _dimensions; j++ ) {
            normalizedData[i][j] = (dataIn[i][j] - _min[j])/(_max[j] - _min[j]);
            //note the normalization by _max - _min in each dimension
        }
    }

    _kdTree.Initialize(normalizedData);
    _pts = _kdTree.getPoints();
    _nFunctions = ff.template getSize < 1 > ();
    _function = allocate(ndarray::makeVector(_pts, _nFunctions));
    _function.deep() = ff;
}


template  < typename T >
T GaussianProcess < T > ::interpolate(ndarray::Array < T,1,1 >  variance,
                                      ndarray::Array < T,1,1 >  const &vin,
                                      int numberOfNeighbors) const
{

    if(_nFunctions > 1){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You need to call the version of GaussianProcess.interpolate() "
                          "that accepts mu and variance arrays (which it populates with results). "
                          "You are interpolating more than one function.");
    }

    if(numberOfNeighbors <= 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for zero or negative number of neighbors\n");
    }

    if(numberOfNeighbors > _kdTree.getPoints()){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for more neighbors than you have data points\n");
    }

    if(variance.getNumElements() != _nFunctions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Your variance array is the incorrect size for the number "
                          "of functions you are trying to interpolate\n");
    }

    if(vin.getNumElements() != _dimensions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You are interpolating at a point with different dimensionality than you data\n");
    }

    int i,j;
    T fbar,mu;

    ndarray::Array < T,1,1 >  covarianceTestPoint;
    ndarray::Array < int,1,1 >  neighbors;
    ndarray::Array < double,1,1 >  neighborDistances,vv;

    Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >  covariance,bb,xx;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;


    _timer.start();

    bb.resize(numberOfNeighbors, 1);
    xx.resize(numberOfNeighbors, 1);

    covariance.resize(numberOfNeighbors,numberOfNeighbors);

    covarianceTestPoint = allocate(ndarray::makeVector(numberOfNeighbors));

    neighbors = allocate(ndarray::makeVector(numberOfNeighbors));

    neighborDistances = allocate(ndarray::makeVector(numberOfNeighbors));

    vv = allocate(ndarray::makeVector(_dimensions));
    if(_useMaxMin == 1){
        //if you constructed this Gaussian process with minimum and maximum
	//values for the dimensions of your parameter space,
        //the point you are interpolating must be scaled to match the data so
	//that the selected nearest neighbors are appropriate

        for(i = 0; i < _dimensions; i++ ) vv[i] = (vin[i] - _min[i])/(_max[i] - _min[i]);
    }
    else{
        vv = vin;
    }


    _kdTree.findNeighbors(neighbors, neighborDistances, vv, numberOfNeighbors);

    _timer.addToSearch();

    fbar = 0.0;
    for(i = 0; i < numberOfNeighbors; i++ )fbar += _function[neighbors[i]][0];
    fbar = fbar/double(numberOfNeighbors);

    for(i = 0; i < numberOfNeighbors; i++ ){
        covarianceTestPoint[i] = (*_covariogram)(vv, _kdTree.getData(neighbors[i]));

        covariance(i,i) = (*_covariogram)(_kdTree.getData(neighbors[i]), _kdTree.getData(neighbors[i]))
                         + _lambda;

        for(j = i + 1; j < numberOfNeighbors; j++ ){
            covariance(i,j) = (*_covariogram)(_kdTree.getData(neighbors[i]),
	                                      _kdTree.getData(neighbors[j]));
            covariance(j,i) = covariance(i, j);
        }
    }

    _timer.addToIteration();

    //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
    ldlt.compute(covariance);

    for(i = 0; i < numberOfNeighbors; i++) bb(i,0) = _function[neighbors[i]][0] - fbar;

    xx = ldlt.solve(bb);
    _timer.addToEigen();

    mu = fbar;

    for(i = 0; i < numberOfNeighbors; i++ ){
        mu += covarianceTestPoint[i]*xx(i, 0);
    }

    _timer.addToIteration();

    variance(0) = (*_covariogram)(vv, vv) + _lambda;

    for(i = 0; i < numberOfNeighbors; i++ ) bb(i) = covarianceTestPoint[i];

    xx = ldlt.solve(bb);

    for(i = 0; i < numberOfNeighbors; i++ ){
        variance(0) -= covarianceTestPoint[i]*xx(i, 0);
    }

    variance(0) = variance(0)*_krigingParameter;

    _timer.addToVariance();
    _timer.addToTotal(1);

    return mu;
}

template  < typename T >
void GaussianProcess < T > ::interpolate(ndarray::Array < T,1,1 >  mu,
                                         ndarray::Array < T,1,1 >  variance,
                                         ndarray::Array < T,1,1 >  const &vin,
                                         int numberOfNeighbors) const
{


    if(numberOfNeighbors <= 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for zero or negative number of neighbors\n");
    }

    if(numberOfNeighbors > _kdTree.getPoints()){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for more neighbors than you have data points\n");
    }

    if(vin.getNumElements() != _dimensions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "You are interpolating at a point with different dimensionality than you data\n");
    }

    if(mu.getNumElements() != _nFunctions || variance.getNumElements() != _nFunctions){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Your mu and/or var arrays are improperly sized for the number of functions "
                          "you are interpolating\n");
    }

    int i,j,ii;
    T fbar;

    ndarray::Array < T,1,1 >  covarianceTestPoint;
    ndarray::Array < int,1,1 >  neighbors;
    ndarray::Array < double,1,1 >  neighborDistances,vv;

    Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >  covariance,bb,xx;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    _timer.start();


     bb.resize(numberOfNeighbors,1);
     xx.resize(numberOfNeighbors,1);
     covariance.resize(numberOfNeighbors,numberOfNeighbors);
     covarianceTestPoint = allocate(ndarray::makeVector(numberOfNeighbors));
     neighbors = allocate(ndarray::makeVector(numberOfNeighbors));
     neighborDistances = allocate(ndarray::makeVector(numberOfNeighbors));

     vv = allocate(ndarray::makeVector(_dimensions));


    if(_useMaxMin == 1) {
        //if you constructed this Gaussian process with minimum and maximum
	//values for the dimensions of your parameter space,
        //the point you are interpolating must be scaled to match the data so
	//that the selected nearest neighbors are appropriate

        for(i = 0; i < _dimensions; i++ )vv[i] = (vin[i] - _min[i])/(_max[i] - _min[i]);
    }
    else {
        vv = vin;
    }


    _kdTree.findNeighbors(neighbors, neighborDistances, vv, numberOfNeighbors);

    _timer.addToSearch();

    for(i = 0; i < numberOfNeighbors; i++ ) {
        covarianceTestPoint[i] = (*_covariogram)(vv, _kdTree.getData(neighbors[i]));

        covariance(i,i) = (*_covariogram)(_kdTree.getData(neighbors[i]), _kdTree.getData(neighbors[i]))
                           + _lambda;

        for(j = i + 1; j < numberOfNeighbors; j++ ){
            covariance(i,j) = (*_covariogram)(_kdTree.getData(neighbors[i]),
	                                      _kdTree.getData(neighbors[j]));
            covariance(j,i) = covariance(i, j);
        }
    }

    _timer.addToIteration();

    //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
    ldlt.compute(covariance);

    for(ii = 0; ii < _nFunctions; ii++ ) {

          fbar = 0.0;
          for(i = 0; i < numberOfNeighbors; i++ )fbar += _function[neighbors[i]][ii];
          fbar = fbar/double(numberOfNeighbors);

          for(i = 0; i < numberOfNeighbors; i++ )bb(i,0) = _function[neighbors[i]][ii] - fbar;
          xx = ldlt.solve(bb);

          mu[ii] = fbar;

          for(i = 0; i < numberOfNeighbors; i++ ) {
            mu[ii] += covarianceTestPoint[i]*xx(i, 0);
          }

    }//ii = 0 through _nFunctions

    _timer.addToEigen();

    variance[0] = (*_covariogram)(vv, vv) + _lambda;

    for(i = 0; i < numberOfNeighbors; i++ )bb(i) = covarianceTestPoint[i];

    xx = ldlt.solve(bb);

    for(i = 0; i < numberOfNeighbors; i++ ) {
        variance[0] -= covarianceTestPoint[i]*xx(i, 0);
    }
    variance[0] = variance[0]*_krigingParameter;


    for(i = 1; i < _nFunctions; i++ )variance[i] = variance[0];

    _timer.addToVariance();
    _timer.addToTotal(1);
}


template  < typename T >
T GaussianProcess < T > ::selfInterpolate(ndarray::Array < T,1,1 >  variance,
                                          int dex,
                                          int numberOfNeighbors) const
{

    if(numberOfNeighbors <= 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for zero or negative number of neighbors\n");
    }

    if(numberOfNeighbors + 1 > _kdTree.getPoints()){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for more neighbors than you have data points\n");
    }

    if(dex < 0 || dex >=_kdTree.getPoints()){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked to self interpolate on a point that does not exist\n");
    }

    int i,j;
    T fbar,mu;

    ndarray::Array < T,1,1 >  covarianceTestPoint;
    ndarray::Array < int,1,1 >  selfNeighbors;
    ndarray::Array < double,1,1 >  selfDistances;
    ndarray::Array < int,1,1 >  neighbors;
    ndarray::Array < double,1,1 >  neighborDistances;

    Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >  covariance,bb,xx;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    _timer.start();

    bb.resize(numberOfNeighbors, 1);
    xx.resize(numberOfNeighbors, 1);
    covariance.resize(numberOfNeighbors,numberOfNeighbors);
    covarianceTestPoint = allocate(ndarray::makeVector(numberOfNeighbors));
    neighbors = allocate(ndarray::makeVector(numberOfNeighbors));
    neighborDistances = allocate(ndarray::makeVector(numberOfNeighbors));

    selfNeighbors = allocate(ndarray::makeVector(numberOfNeighbors + 1));
    selfDistances = allocate(ndarray::makeVector(numberOfNeighbors + 1));

    //we don't use _useMaxMin because the data has already been normalized


    _kdTree.findNeighbors(selfNeighbors, selfDistances, _kdTree.getData(dex),
                          numberOfNeighbors + 1);

    _timer.addToSearch();

    if(selfNeighbors[0]!= dex) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
        "Nearest neighbor search in selfInterpolate did not find self\n");
    }

    //SelfNeighbors[0] will be the point itself (it is its own nearest neighbor)
    //We discard that for the interpolation calculation
    //
    //If you do not wish to do this, simply call the usual ::interpolate() method instead of
    //::selfInterpolate()
    for(i = 0; i < numberOfNeighbors; i++ ){
        neighbors[i] = selfNeighbors[i + 1];
        neighborDistances[i] = selfDistances[i + 1];
    }

    fbar = 0.0;
    for(i = 0; i < numberOfNeighbors; i++ ) fbar += _function[neighbors[i]][0];
    fbar = fbar/double(numberOfNeighbors);

    for(i = 0; i < numberOfNeighbors; i++ ){
        covarianceTestPoint[i] = (*_covariogram)(_kdTree.getData(dex),
                                                 _kdTree.getData(neighbors[i]));

        covariance(i, i) = (*_covariogram)(_kdTree.getData(neighbors[i]),
                                           _kdTree.getData(neighbors[i])) + _lambda;

      for(j = i + 1; j < numberOfNeighbors; j++ ) {
          covariance(i, j) = (*_covariogram)(_kdTree.getData(neighbors[i]),
                                             _kdTree.getData(neighbors[j]));
          covariance(j, i) = covariance(i ,j);
        }
    }
    _timer.addToIteration();

    //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
    ldlt.compute(covariance);


    for(i = 0; i < numberOfNeighbors;i++ ) bb(i, 0) = _function[neighbors[i]][0] - fbar;
    xx = ldlt.solve(bb);
    _timer.addToEigen();

    mu = fbar;

    for(i = 0; i < numberOfNeighbors; i++ ) {
        mu += covarianceTestPoint[i]*xx(i, 0);
    }


    variance(0) = (*_covariogram)(_kdTree.getData(dex), _kdTree.getData(dex)) + _lambda;

    for(i = 0; i < numberOfNeighbors; i++ )bb(i) = covarianceTestPoint[i];

    xx = ldlt.solve(bb);

    for(i = 0; i < numberOfNeighbors; i++ ){
        variance(0) -= covarianceTestPoint[i]*xx(i,0);
    }

    variance(0) = variance(0)*_krigingParameter;
    _timer.addToVariance();
    _timer.addToTotal(1);

    return mu;
}

template  < typename T >
void GaussianProcess < T > ::selfInterpolate(ndarray::Array < T,1,1 >  mu,
                                             ndarray::Array < T,1,1 >  variance,
                                             int dex,
                                             int numberOfNeighbors) const{

    if(numberOfNeighbors <= 0){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for zero or negative number of neighbors\n");
    }

    if(numberOfNeighbors + 1 > _kdTree.getPoints()){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked for more neighbors than you have data points\n");
    }

    if(dex < 0 || dex >=_kdTree.getPoints()){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "Asked to self interpolate on a point that does not exist\n");
    }

    int i,j,ii;
    T fbar;

    ndarray::Array < T,1,1 >  covarianceTestPoint;
    ndarray::Array < int,1,1 >  selfNeighbors;
    ndarray::Array < double,1,1 >  selfDistances;
    ndarray::Array < int,1,1 >  neighbors;
    ndarray::Array < double,1,1 >  neighborDistances;

    Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >  covariance,bb,xx;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    _timer.start();

    bb.resize(numberOfNeighbors, 1);
    xx.resize(numberOfNeighbors, 1);
    covariance.resize(numberOfNeighbors,numberOfNeighbors);
    covarianceTestPoint = allocate(ndarray::makeVector(numberOfNeighbors));
    neighbors = allocate(ndarray::makeVector(numberOfNeighbors));
    neighborDistances = allocate(ndarray::makeVector(numberOfNeighbors));

    selfNeighbors = allocate(ndarray::makeVector(numberOfNeighbors + 1));
    selfDistances = allocate(ndarray::makeVector(numberOfNeighbors + 1));

    //we don't use _useMaxMin because the data has already been normalized


    _kdTree.findNeighbors(selfNeighbors, selfDistances, _kdTree.getData(dex),
                          numberOfNeighbors + 1);

    _timer.addToSearch();

    if(selfNeighbors[0]!= dex) {

        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
        "Nearest neighbor search in selfInterpolate did not find self\n");
    }

    //SelfNeighbors[0] will be the point itself (it is its own nearest neighbor)
    //We discard that for the interpolation calculation
    //
    //If you do not wish to do this, simply call the usual ::interpolate() method instead of
    //::selfInterpolate()
    for(i = 0; i < numberOfNeighbors; i++ ) {
        neighbors[i] = selfNeighbors[i + 1];
        neighborDistances[i] = selfDistances[i + 1];
    }



    for(i = 0; i < numberOfNeighbors; i++ ) {
        covarianceTestPoint[i] = (*_covariogram)(_kdTree.getData(dex),_kdTree.getData(neighbors[i]));

        covariance(i, i) = (*_covariogram)(_kdTree.getData(neighbors[i]), _kdTree.getData(neighbors[i]))
                           + _lambda;

        for(j = i + 1; j < numberOfNeighbors; j++ ) {
            covariance(i, j) = (*_covariogram)(_kdTree.getData(neighbors[i]),
                                               _kdTree.getData(neighbors[j]));
            covariance(j, i) = covariance(i, j);
        }
    }
    _timer.addToIteration();


    //use Eigen's ldlt solver in place of matrix inversion (for speed purposes)
    ldlt.compute(covariance);

    for(ii = 0; ii < _nFunctions; ii++ ) {

        fbar = 0.0;
        for(i = 0; i < numberOfNeighbors; i++ )fbar += _function[neighbors[i]][ii];
        fbar = fbar/double(numberOfNeighbors);

        for(i = 0; i < numberOfNeighbors; i++ )bb(i,0) = _function[neighbors[i]][ii] - fbar;
        xx = ldlt.solve(bb);

        mu[ii] = fbar;

        for(i = 0; i < numberOfNeighbors; i++ ){
            mu[ii] += covarianceTestPoint[i]*xx(i,0);
        }
    }//ii = 0 through _nFunctions

    _timer.addToEigen();

    variance[0] = (*_covariogram)(_kdTree.getData(dex), _kdTree.getData(dex)) + _lambda;

    for(i = 0; i < numberOfNeighbors; i++ )bb(i) = covarianceTestPoint[i];

    xx = ldlt.solve(bb);

    for(i = 0; i < numberOfNeighbors; i++ ) {
        variance[0] -= covarianceTestPoint[i]*xx(i,0);
    }

    variance[0] = variance[0]*_krigingParameter;

    for(i = 1; i < _nFunctions; i++ )variance[i] = variance[0];

    _timer.addToVariance();
    _timer.addToTotal(1);
}


template < typename T >
void GaussianProcess < T > ::batchInterpolate(ndarray::Array < T,1,1 >  mu,
                                              ndarray:: Array < T,1,1 >  variance,
                                              ndarray::Array < T,2,2 >  const &queries) const
{

    int i,j,ii,nQueries;

    T fbar;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  batchCovariance,batchbb,batchxx;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  queryCovariance;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    ndarray::Array < T,1,1 >  v1;

    _timer.start();

    nQueries = queries.template getSize < 0 > ();


    v1 = allocate(ndarray::makeVector(_dimensions));
    batchbb.resize(_pts, 1);
    batchxx.resize(_pts, 1);
    batchCovariance.resize(_pts, _pts);
    queryCovariance.resize(_pts, 1);


    for(i = 0; i < _pts; i++ ) {

        batchCovariance(i, i) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(i)) + _lambda;
        for(j = i + 1; j < _pts; j++ ) {
            batchCovariance(i, j) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(j));
            batchCovariance(j, i) = batchCovariance(i, j);
        }
    }
    _timer.addToIteration();

    ldlt.compute(batchCovariance);

    fbar = 0.0;
    for(i = 0; i < _pts; i++ ) {
        fbar += _function[i][0];
    }
    fbar = fbar/T(_pts);

    for(i = 0; i < _pts; i++ ){
        batchbb(i, 0) = _function[i][0] - fbar;
    }
    batchxx = ldlt.solve(batchbb);
    _timer.addToEigen();


    for(ii = 0; ii < nQueries; ii++ ) {
        for(i = 0; i < _dimensions; i++ )v1[i] = queries[ii][i];
        if(_useMaxMin == 1) {
            for(i = 0; i < _dimensions; i++ )v1[i] = (v1[i] - _min[i])/(_max[i] - _min[i]);
        }
        mu(ii) = fbar;
        for(i = 0; i < _pts; i++ ){
            mu(ii) += batchxx(i)*(*_covariogram)(v1, _kdTree.getData(i));
        }
    }
    _timer.addToIteration();

    for(ii = 0; ii < nQueries; ii++ ) {
        for(i = 0; i < _dimensions; i++ )v1[i] = queries[ii][i];
        if(_useMaxMin == 1) {
            for(i = 0; i < _dimensions; i++ )v1[i] = (v1[i] - _min[i])/(_max[i] - _min[i]);
        }

        for(i = 0; i < _pts; i++ ) {
            batchbb(i, 0) = (*_covariogram)(v1, _kdTree.getData(i));
            queryCovariance(i, 0) = batchbb(i, 0);
        }
        batchxx = ldlt.solve(batchbb);

        variance[ii] = (*_covariogram)(v1, v1) + _lambda;

        for(i = 0; i < _pts; i++ ){
            variance[ii] -= queryCovariance(i, 0)*batchxx(i);
        }

        variance[ii] = variance[ii]*_krigingParameter;

    }

    _timer.addToVariance();
    _timer.addToTotal(nQueries);
}

template < typename T >
void GaussianProcess < T > ::batchInterpolate(ndarray::Array < T,2,2 >  mu,
                                              ndarray:: Array < T,2,2 >  variance,
                                              ndarray::Array < T,2,2 >  const &queries) const
{

    int i,j,ii,nQueries,ifn;

    T fbar;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  batchCovariance,batchbb,batchxx;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  queryCovariance;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    ndarray::Array < T,1,1 >  v1;

    _timer.start();

    nQueries = queries.template getSize < 0 > ();



    v1 = allocate(ndarray::makeVector(_dimensions));
    batchbb.resize(_pts, 1);
    batchxx.resize(_pts, 1);
    batchCovariance.resize(_pts, _pts);
    queryCovariance.resize(_pts, 1);

    for(i = 0; i < _pts; i++ ){
        batchCovariance(i, i) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(i)) + _lambda;
        for(j = i + 1; j < _pts; j++ ) {
            batchCovariance(i, j) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(j));
            batchCovariance(j, i) = batchCovariance(i, j);
        }
    }

    _timer.addToIteration();

    ldlt.compute(batchCovariance);

    _timer.addToEigen();

    for(ifn = 0; ifn < _nFunctions; ifn++ ) {

        fbar = 0.0;
        for(i = 0; i < _pts; i++ ){
            fbar += _function[i][ifn];
        }
        fbar = fbar/T(_pts);
        _timer.addToIteration();

        for(i = 0; i < _pts; i++ ){
            batchbb(i,0) = _function[i][ifn] - fbar;
        }
        batchxx = ldlt.solve(batchbb);
        _timer.addToEigen();


        for(ii = 0; ii < nQueries; ii++ ){
            for(i = 0; i < _dimensions; i++ ) v1[i] = queries[ii][i];
            if(_useMaxMin == 1) {
                for(i = 0; i < _dimensions; i++ ) v1[i] = (v1[i] - _min[i])/(_max[i] - _min[i]);
            }
            mu[ii][ifn] = fbar;
            for(i = 0; i < _pts; i++ ){
                mu[ii][ifn] += batchxx(i)*(*_covariogram)(v1, _kdTree.getData(i));
            }
        }

    }//ifn = 0 to _nFunctions

    _timer.addToIteration();
    for(ii = 0; ii < nQueries; ii++ ){
        for(i = 0; i < _dimensions; i++ ) v1[i] = queries[ii][i];
        if(_useMaxMin == 1){
            for(i = 0; i < _dimensions; i++ ) v1[i] = (v1[i] - _min[i])/(_max[i] - _min[i]);
        }

        for(i = 0;i < _pts;i++ ) {
            batchbb(i,0) = (*_covariogram)(v1,_kdTree.getData(i));
            queryCovariance(i,0) = batchbb(i,0);
        }
        batchxx = ldlt.solve(batchbb);

        variance[ii][0] = (*_covariogram)(v1, v1) + _lambda;

        for(i = 0; i < _pts; i++ ) {
              variance[ii][0] -= queryCovariance(i, 0)*batchxx(i);
        }

        variance[ii][0] = variance[ii][0]*_krigingParameter;
        for(i = 1; i < _nFunctions; i++ ) variance[ii][i] = variance[ii][0];

    }
    _timer.addToVariance();
    _timer.addToTotal(nQueries);
}

template < typename T >
void GaussianProcess < T > ::batchInterpolate(ndarray::Array < T,1,1 >  mu,
                                              ndarray::Array < T,2,2 >  const &queries) const
{

    int i,j,ii,nQueries;

    T fbar;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  batchCovariance,batchbb,batchxx;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  queryCovariance;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    ndarray::Array < T,1,1 >  v1;

    _timer.start();

    nQueries = queries.template getSize < 0 > ();

    v1 = allocate(ndarray::makeVector(_dimensions));

    batchbb.resize(_pts, 1);
    batchxx.resize(_pts, 1);
    batchCovariance.resize(_pts, _pts);
    queryCovariance.resize(_pts, 1);


    for(i = 0; i < _pts; i++ ) {
        batchCovariance(i, i) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(i)) + _lambda;
        for(j = i + 1; j < _pts; j++ ) {
            batchCovariance(i, j) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(j));
            batchCovariance(j, i) = batchCovariance(i, j);
        }
    }
    _timer.addToIteration();

    ldlt.compute(batchCovariance);

    fbar = 0.0;
    for(i = 0; i < _pts; i++ ) {
        fbar += _function[i][0];
    }
    fbar = fbar/T(_pts);

    for(i = 0; i < _pts; i++ ) {
        batchbb(i, 0) = _function[i][0] - fbar;
    }
    batchxx = ldlt.solve(batchbb);
    _timer.addToEigen();

    for(ii = 0; ii < nQueries; ii++ ) {

        for(i = 0; i < _dimensions; i++ ) v1[i] = queries[ii][i];
        if(_useMaxMin == 1) {
            for(i = 0; i < _dimensions; i++ )v1[i] = (v1[i] - _min[i])/(_max[i] - _min[i]);
        }

        mu(ii) = fbar;
        for(i = 0; i < _pts; i++ ) {
            mu(ii) += batchxx(i)*(*_covariogram)(v1, _kdTree.getData(i));
        }
    }
    _timer.addToIteration();
    _timer.addToTotal(nQueries);

}

template < typename T >
void GaussianProcess < T > ::batchInterpolate(ndarray::Array < T,2,2 >  mu,
                                              ndarray::Array < T,2,2 >  const &queries) const
{

    int i,j,ii,nQueries,ifn;


    T fbar;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  batchCovariance,batchbb,batchxx;
    Eigen::Matrix  < T,Eigen::Dynamic,Eigen::Dynamic >  queryCovariance;
    Eigen::LDLT < Eigen::Matrix < T,Eigen::Dynamic,Eigen::Dynamic >   >  ldlt;

    ndarray::Array < T,1,1 >  v1;

    _timer.start();

    nQueries = queries.template getSize < 0 > ();


    v1 = allocate(ndarray::makeVector(_dimensions));

    batchbb.resize(_pts, 1);
    batchxx.resize(_pts, 1);
    batchCovariance.resize(_pts, _pts);
    queryCovariance.resize(_pts, 1);


    for(i = 0; i < _pts; i++ ){
        batchCovariance(i, i) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(i)) + _lambda;
        for(j = i + 1; j < _pts; j++ ){
            batchCovariance(i, j) = (*_covariogram)(_kdTree.getData(i), _kdTree.getData(j));
            batchCovariance(j, i) = batchCovariance(i, j);
        }
    }

    _timer.addToIteration();

    ldlt.compute(batchCovariance);

    _timer.addToEigen();

    for(ifn = 0; ifn < _nFunctions; ifn++ ){

        fbar = 0.0;
        for(i = 0; i < _pts; i++ ){
            fbar += _function[i][ifn];
        }
        fbar = fbar/T(_pts);

        _timer.addToIteration();

        for(i = 0; i < _pts; i++ ){
            batchbb(i, 0) = _function[i][ifn] - fbar;
        }
        batchxx = ldlt.solve(batchbb);
        _timer.addToEigen();


        for(ii = 0; ii < nQueries; ii++ ){
            for(i = 0; i < _dimensions; i++ )v1[i] = queries[ii][i];
            if(_useMaxMin == 1){
                for(i = 0;i < _dimensions;i++ )v1[i] = (v1[i] - _min[i])/(_max[i] - _min[i]);
            }

            mu[ii][ifn] = fbar;
            for(i = 0; i < _pts; i++ ){
                 mu[ii][ifn] += batchxx(i)*(*_covariogram)(v1, _kdTree.getData(i));
            }
        }


    }//ifn = 0 through _nFunctions

    _timer.addToTotal(nQueries);

}


template  < typename T >
void GaussianProcess < T > ::addPoint(ndarray::Array < T,1,1 >  const &vin, T f)
{

    int i,j;

    if(_nFunctions!= 1){

        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
        "You are calling the wrong addPoint; you need a vector of functions\n");

    }

    ndarray::Array < T,1,1 >  v;
    v = allocate(ndarray::makeVector(_dimensions));

    for(i = 0; i < _dimensions; i++ ){
        v[i] = vin[i];
        if(_useMaxMin == 1){
            v[i] = (v[i] - _min[i])/(_max[i] - _min[i]);
        }
    }

    if(_pts == _room){
        ndarray::Array < T,2,2 >  buff;
        buff = allocate(ndarray::makeVector(_pts, _nFunctions));
        buff.deep() = _function;

        _room += _roomStep;
        _function = allocate(ndarray::makeVector(_room, _nFunctions));
        for(i = 0; i < _pts; i++ ) {

            for(j = 0; j < _nFunctions; j++ ) {
                _function[i][j] = buff[i][j];
            }

        }
    }

    _function[_pts][0] = f;

    _kdTree.addPoint(v);
    _pts = _kdTree.getPoints();

}

template  < typename T >
void GaussianProcess < T > ::addPoint(ndarray::Array < T,1,1 >  const &vin,
                                      ndarray::Array < T,1,1 >  const &f)
{

    int i,j;

    ndarray::Array < T,1,1 >  v;
    v = allocate(ndarray::makeVector(_dimensions));

    for(i = 0; i < _dimensions; i++ ) {
        v[i] = vin[i];
        if(_useMaxMin == 1) {
            v[i] = (v[i] - _min[i])/(_max[i] - _min[i]);
        }
    }

    if(_pts == _room) {
        ndarray::Array < T,2,2 >  buff;
        buff = allocate(ndarray::makeVector(_pts, _nFunctions));
        buff.deep() = _function;

        _room += _roomStep;
        _function = allocate(ndarray::makeVector(_room, _nFunctions));
        for(i = 0; i < _pts; i++ ) {
            for(j = 0; j < _nFunctions; j++ ) {
                 _function[i][j] = buff[i][j];
            }
        }

    }
    for(i = 0; i < _nFunctions; i++ )_function[_pts][i] = f[i];

    _kdTree.addPoint(v);
    _pts = _kdTree.getPoints();


}

template  < typename T >
void GaussianProcess < T > ::removePoint(int dex)
{

    int i,j;

    _kdTree.removePoint(dex);

    for(i = dex; i < _pts; i++ ) {
        for(j = 0; j < _nFunctions; j++ ) {
             _function[i][j] = _function[i + 1][j];
        }
    }
    _pts = _kdTree.getPoints();
}

template  < typename T >
void GaussianProcess < T > ::setKrigingParameter(T kk)
{
    _krigingParameter = kk;
}

template  < typename T >
void GaussianProcess < T > ::setCovariogram(std::shared_ptr <  Covariogram < T >   >  const &covar){
     _covariogram = covar;
}

template  < typename T >
void GaussianProcess < T > ::setLambda(T lambda){
    _lambda = lambda;
}


template  < typename T >
GaussianProcessTimer& GaussianProcess < T > ::getTimes() const
{
    return _timer;
}

template  < typename T >
Covariogram < T > ::~Covariogram(){};

template  < typename T >
T Covariogram < T > ::operator()(ndarray::Array < const T,1,1 >  const &p1,
                                 ndarray::Array < const T,1,1 >  const &p2) const
{
    std::cout << "by the way, you are calling the wrong operator\n";
    exit(1);
    return T(1.0);
}

template  < typename T >
SquaredExpCovariogram < T > ::~SquaredExpCovariogram(){}

template  < typename T >
SquaredExpCovariogram < T > ::SquaredExpCovariogram()
{
    _ellSquared = 1.0;
}

template  < typename T >
void SquaredExpCovariogram < T > ::setEllSquared(double ellSquared)
{
    _ellSquared = ellSquared;
}

template  < typename T >
T SquaredExpCovariogram < T > ::operator()(ndarray::Array < const T,1,1 >  const &p1,
                                           ndarray::Array < const T,1,1 >  const &p2) const
{
    int i;
    T d;
    d = 0.0;
    for(i = 0; i < p1.template getSize < 0 > (); i++ ){
        d += (p1[i] - p2[i])*(p1[i] - p2[i]);
    }

    d = d/_ellSquared;
    return T(exp( - 0.5*d));
}

template  < typename T >
NeuralNetCovariogram < T > ::~NeuralNetCovariogram(){}

template  < typename T >
NeuralNetCovariogram < T > ::NeuralNetCovariogram(){

    _sigma0 = 1.0;
    _sigma1 = 1.0;
}

template  < typename T >
T NeuralNetCovariogram < T > ::operator()(ndarray::Array < const T,1,1 >  const &p1,
                                          ndarray::Array < const T,1,1 >  const &p2
                                          ) const
{
    int i,dim;
    double num,denom1,denom2,arg;

    dim = p1.template getSize < 0 > ();

    num = 2.0*_sigma0;
    denom1 = 1.0 + 2.0*_sigma0;
    denom2 = 1.0 + 2.0*_sigma0;

    for(i = 0; i < dim; i++ ) {
        num += 2.0*p1[i]*p2[i]*_sigma1;
        denom1 += 2.0*p1[i]*p1[i]*_sigma1;
        denom2 += 2.0*p2[i]*p2[i]*_sigma1;
    }

    arg = num/::sqrt(denom1*denom2);
    return T(2.0*(::asin(arg))/3.141592654);

}

template  < typename T >
void NeuralNetCovariogram < T > ::setSigma0(double sigma0)
{
    _sigma0 = sigma0;
}

template  < typename T >
void NeuralNetCovariogram < T > ::setSigma1(double sigma1)
{
    _sigma1 = sigma1;
}

}}}

#define gpn lsst::afw::math

#define INSTANTIATEGP(T) \
        template class gpn::KdTree < T > ; \
        template class gpn::GaussianProcess < T > ; \
        template class gpn::Covariogram < T > ; \
        template class gpn::SquaredExpCovariogram < T > ;\
        template class gpn::NeuralNetCovariogram < T > ;

INSTANTIATEGP(double);
