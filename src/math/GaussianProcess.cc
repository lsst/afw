#include "lsst/afw/math/GaussianProcess.h"
//#include "gptest/gptest.h"
#include <iostream>
#include <cmath>

using namespace std;

namespace gptest{

namespace GaussianProcessFunctions{

template <typename datatype>
int mergeScanner(datatype *m, int *indices, int dex, int el){
/*this will take the matrix m and put everything in it with value
greater than element m[dex] to the right of that element
and everything less than m[dex] to the left; it then returns
the new index of that anchored element (which you now *know* is in
the right spot*/

/*this is an implemenation of the merge sort algorithm described in
Numerical Recipes (2nd edition); Press, Teukolsky, Vetterling, and
Flannery 1992*/
  
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
  
  //now what was m[dex] is in the right place
  
  
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

template <typename datatype>
void mergeSort(datatype *insort, int *indices, int el){
  
  int i,k;
  datatype nn;
  
  //printf("\nin sort\n");
  //for(i=0;i<el;i++)printf("%e\n",insort[i]);
  //printf("\n");
  
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


template<typename dty>
double euclideanDistance(dty *v1, dty *v2, int d_dim){
  int i;
  double dd;
  dd=0.0;
  for(i=0;i<d_dim;i++){
    dd+=double(v1[i]-v2[i])*double(v1[i]-v2[i]);
  }
  
  return sqrt(dd);
}

template<typename dtyi, typename dtyo>
dtyo expCovariogram(dtyi *v1, dtyi *v2, int d_dim, double *hyp){
  double dd;
  dd=euclideanDistance(v1,v2,d_dim);
  return dtyo(exp(-0.5*dd*dd/hyp[0]));
}

}


namespace GPfn = gptest::GaussianProcessFunctions;

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

template <typename datatype>
KdTree<datatype>::KdTree(int dd, int pp, datatype **dt, \
double(*dfn)(datatype*,datatype*,int)){
  
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

template <typename datatype>
void KdTree<datatype>::_organize(int *use, int ct, int parent, int dir){
  //*use is the list of data indices that we must organize
  //ct is the number of indices available
  //parent is the parent node of the daughter this call will generate
  //dir denotes which side of the parent we have descended on
  
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

template <typename datatype>
void KdTree<datatype>::getTreeNode(int dex, int *v){
  v[0]=_tree[dex][0];
  v[1]=_tree[dex][1];
  v[2]=_tree[dex][2];
  v[3]=_tree[dex][3];
}


template <typename datatype>
void KdTree<datatype>::testScanner(){
   
   datatype *mm,*mmshld;
   int i,*innt,*innshld;
   
   mm=new datatype[5];
   innt=new int[5];
   
   mmshld=new datatype[5];
   innshld=new int[5];
   
   mm[0]=datatype(2.0);
   mm[1]=datatype(4.0);
   mm[2]=datatype(-2.0);
   mm[3]=datatype(3.0);
   mm[4]=datatype(2.0);
   
   innt[0]=1;
   innt[1]=-2;
   innt[2]=4;
   innt[3]=5;
   innt[4]=3;
   
   mmshld[0]=datatype(-2.0);
   mmshld[1]=datatype(2.0);
   mmshld[2]=datatype(2.0);
   mmshld[3]=datatype(3.0);
   mmshld[4]=datatype(4.0);

   innshld[0]=4;
   innshld[1]=3;
   innshld[2]=1;
   innshld[3]=5;
   innshld[4]=-2;
   
   i=GPfn::mergeScanner<datatype>(mm,innt,4,5);
   
   if(i!=1){
     std::cout<<"FAILED _mergeScanner i "<<i<<"\n";;
   }
   
   for(i=0;i<5;i++){
     if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
      std::cout<<"FAILED _mergeScanner "<<i<<"is "<<innt[i]<<" shld "<<innshld[i]<<"\n"; 
     }
   }
   
   
   mm[0]=datatype(2.0);
   mm[1]=datatype(2.0);
   mm[2]=datatype(2.0);
   mm[3]=datatype(2.0);
   mm[4]=datatype(2.0);
   
   innt[0]=1;
   innt[1]=-2;
   innt[2]=4;
   innt[3]=5;
   innt[4]=3;
   
   mmshld[0]=datatype(2.0);
   mmshld[1]=datatype(2.0);
   mmshld[2]=datatype(2.0);
   mmshld[3]=datatype(2.0);
   mmshld[4]=datatype(2.0);

   innshld[0]=4;
   innshld[1]=-2;
   innshld[2]=1;
   innshld[3]=5;
   innshld[4]=3;
   
   i=GPfn::mergeScanner<datatype>(mm,innt,2,5);
   
   if(i!=0){
     std::cout<<"FAILED homog _mergeScanner i "<<i<<"\n";
   }
   
   for(i=0;i<5;i++){
     if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
       std::cout<<"FAILED homog _mergeScanner "<<i<<" is "<<innt[i]<<" shld "<<innshld[i]<<"\n";
     }
   }
   
    mm[0]=datatype(2.0);
   mm[1]=datatype(3.0);
   mm[2]=datatype(2.0);
   mm[3]=datatype(2.0);
   mm[4]=datatype(2.0);
   
   innt[0]=1;
   innt[1]=-2;
   innt[2]=4;
   innt[3]=5;
   innt[4]=3;
   
   mmshld[0]=datatype(2.0);
   mmshld[1]=datatype(2.0);
   mmshld[2]=datatype(2.0);
   mmshld[3]=datatype(2.0);
   mmshld[4]=datatype(3.0);

   innshld[0]=1;
   innshld[1]=3;
   innshld[2]=4;
   innshld[3]=5;
   innshld[4]=-2;
   
   i=GPfn::mergeScanner<datatype>(mm,innt,1,5);
   
   if(i!=4){
     std::cout<<"FAILED homog_b _mergeScanner i "<<i<<"\n";
   }
   
   for(i=0;i<5;i++){
     if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
       std::cout<<"FAILED homog_b _mergeScanner "<<i<<" is "<<innt[i]<<" shld "<<innshld[i]<<"\n";
     }
   }
   
   delete [] mm;
   delete [] innt;
   delete [] mmshld;
   delete [] innshld;
   
}

template <typename datatype>
void KdTree<datatype>::testSort(){
  
  datatype *mm,*mmshld;
  int *innt,*innshld,i;
  
  mm=new datatype[5];
  innt=new int[5];
  mmshld=new datatype[5];
  innshld=new int[5];
  
  mm[0]=datatype(3.0);
  mm[1]=datatype(1.0);
  mm[2]=datatype(-3.0);
  mm[3]=datatype(-2.0);
  mm[4]=datatype(2.0);
  
  mmshld[0]=datatype(-3.0);
  mmshld[1]=datatype(-2.0);
  mmshld[2]=datatype(1.0);
  mmshld[3]=datatype(2.0);
  mmshld[4]=datatype(3.0);
  
  innshld[0]=2;
  innshld[1]=3;
  innshld[2]=1;
  innshld[3]=4;
  innshld[4]=0;
  
  for(i=0;i<5;i++){
    innt[i]=i;
  }
  
  GPfn::mergeSort<datatype>(mm,innt,5);
  
  for(i=0;i<5;i++){
    //printf("%e %e\n",mm[i],mmshld[i]);
    if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
      std::cout<<"sort FAILED "<<i<<" is "<<innt[i]<<" shld "<<innshld[i]<<"\n";
    }
  }
  
  mm[0]=datatype(1.0);
  mm[1]=datatype(1.0);
  mm[2]=datatype(1.0);
  mm[3]=datatype(1.0);
  mm[4]=datatype(-1.0);
  
  mmshld[0]=datatype(-1.0);
  mmshld[1]=datatype(1.0);
  mmshld[2]=datatype(1.0);
  mmshld[3]=datatype(1.0);
  mmshld[4]=datatype(1.0);
  
  GPfn::mergeSort<datatype>(mm,innt,5);
  for(i=0;i<5;i++){
    if(mm[i]!=mmshld[i]){
      std::cout<<"sort homog FAILED "<<i<<"\n";
    }
  }
  
  delete [] mm;
  delete [] innt;
  delete [] mmshld;
  delete [] innshld;
  
}

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
      
      //printf("_tree FAILURE root %d target %d dir %d %d < %d\n",
      //root,target,dir,data[root][_tree[target][0]],data[target][_tree[target][0]]);
    }
  }
  
  if(_tree[target][3]>=0){
    if(_tree[_tree[target][3]][1]==target)i=1;
    else i=2;
    
    output=output*_walkUpTree(_tree[target][3],i,root);
  
  }
  else{
    output=output*target;
    //so that it will presumably return _masterParent
    //make sure everything is connected to _masterParent
  }
  return output;
  
}

template <typename datatype>
int KdTree<datatype>::testTree(){

  int i,j,*isparent,output;
  
  /*for(i=0;i<_pts;i++){
    for(j=0;j<_dimensions;j++)printf("%d ",data[i][j]);
    printf("\n");
  }
  printf("\n\n");*/
  
  j=0;
  for(i=0;i<_pts;i++){
    if(_tree[i][3]<0)j++;
  }
  if(j!=1){
    std::cout<<"_tree FAILURE "<<j<<" _masterParents\n";
    return 0;
    //printf("_tree FAILURE %d _masterParents\n");
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
     //printf("_tree FAILURE %d is parent to %d\n",i,isparent[i]);
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
  //printf("done with black box test\n");
  std::cout<<"done with black box test of KdTree\n";
  if(output!=_masterParent) return 0;
  else return 1;
}

template <typename datatype>
int KdTree<datatype>::_findNode(datatype *v){
  //this routine finds the node that would be the parent of
  //v[] if you were going to add it to the tree
  
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

template <typename datatype>
void KdTree<datatype>::addPoint(datatype *v){

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

template<typename datatype>
void KdTree<datatype>::findNeighbors(datatype *v, int n_nn, int *neighdex, double *dd){
  //this will search the tree for the n_nn nearest neighbors of v[];
  //the indexes of the neighbors will be stored in neighdex[];
  //the distances will be stored in dd[];
  
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

template<typename datatype>
void KdTree<datatype>::_lookForNeighbors(datatype *v, int consider, int from){
  //in the process of searching for nearest neighbors of v[]
  //this routine will look at the point denoted by 'consider'
  //and walk up/down the tree based on where it is coming from ('from')
  
  int i,j,going;
  double dd;
  
  
  
  dd=_distance(data[consider],v,_dimensions);
  
  /*for(i=1;i<_neighborsFound;i++){
    if(_neighborDistances[i]<_neighborDistances[i-1]){
      printf("_neighborsFound out of order\n");
      exit(1);
    }
  }*/
  
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


/*
template <typename dtyi, typename dtyo>
GaussianProcess<dtyi,dtyo>::GaussianProcess(int dd, int pp, dtyi **datain, dtyi *mx, dtyi *mn, dtyo *ff){
 //constructor if you have maxs and mins
  
  int i,j;
  
  printf("WARNING this constructor is not actually ready\n");
  exit(1);
  

  
  _dimensions=dd;
  _pts=pp;
  _room=_pts;
  
  _function=ff;
  
  _krigingParameter=dtyo(1.0);
  


  
  _max=mx;
  _min=mn;
  
  _covariogram=GPfn::expCovariogram;
  _distance=GPfn::euclideanDistance;
  
  _calledInterpolate=0;

 _lambda=dtyo(1.0e-5);
  
  
  _useMaxMin=1;
  _data=new dtyi*[_pts];
  for(i=0;i<_pts;i++){
    _data[i]=new dtyi[_dimensions];
    for(j=0;j<_dimensions;j++){
      _data[i][j]=(datain[i][j]-_min[j])/(_max[j]-_min[j]);
    }
  }
   
  _kdTreePtr=new KdTree<dtyi>(_dimensions,_pts,_data,_distance);
  
}

*/

template <typename dtyi, typename dtyo>
GaussianProcess<dtyi,dtyo>::GaussianProcess(int dd, int pp, ndarray::Array<dtyi,2,2> datain, \
ndarray::Array<dtyo,1,1> ff){
 //constructor if you do not have maxs and mins
  
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



template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setKrigingParameter(int kk){
  
  Eigen::Matrix <dtyo,Eigen::Dynamic,Eigen::Dynamic> kgg,kggin;
  
  int *kneigh,i,j,k,*inn;
  double *ddneigh;
  dtyo *kggq,mu,sig2,fbar,*rat;

  kneigh=new int[kk+1];
  ddneigh=new double[kk+1];
  kggq=new dtyo[kk];
  
  inn=new int[_pts];
  rat=new dtyo[_pts];
  
  kgg.resize(kk,kk);
  
  for(i=0;i<_pts;i++){

    _kdTreePtr->findNeighbors(_data[i],kk+1,kneigh,ddneigh);
    
    for(j=0;j<kk;j++){
      kggq[j]=_covariogram(_data[i],_data[kneigh[j+1]],_dimensions,_hyperParameters);
      
      kgg(j,j)=_covariogram(_data[kneigh[j+1]],_data[kneigh[j+1]],_dimensions,_hyperParameters)\
      +_lambda;
      
      for(k=j+1;k<kk;k++){
        kgg(j,k)=_covariogram(_data[kneigh[j+1]],_data[kneigh[k+1]],_dimensions,_hyperParameters);
        kgg(k,j)=kgg(j,k);
      }
    }
     
    kggin=kgg.inverse();
    
    fbar=dtyo(0.0);
    for(j=0;j<kk;j++){
      fbar+=_function[kneigh[j+1]];
    }
    fbar=fbar/dtyo(kk);
    
    mu=fbar;
    for(j=0;j<kk;j++){
      for(k=0;k<kk;k++){
        mu+=kggq[j]*kggin(j,k)*(_function[kneigh[k+1]]-fbar);
      }
    }
    
    sig2=_covariogram(_data[i],_data[i],_dimensions,_hyperParameters)+_lambda;
    
    for(j=0;j<kk;j++){
      sig2-=kggq[j]*kggin(j,j)*kggq[j];
      for(k=j+1;k<kk;k++){
         sig2-=2.0*kggq[j]*kggin(j,k)*kggq[k];
      }
    }
    
    rat[i]=(mu-_function[i])*(mu-_function[i])/sig2; 
    
  }
  
  GPfn::mergeSort<dtyo>(rat,inn,_pts);
  
  _krigingParameter=rat[68*_pts/100];
  
  delete [] kneigh;
  delete [] ddneigh;
  delete [] kggq;
  delete [] inn;
  delete [] rat;
  
  
}


template <typename dtyi, typename dtyo>
dtyo GaussianProcess<dtyi,dtyo>::interpolate(ndarray::Array<dtyi,1,1> vin, ndarray::Array<dtyo,1,1> variance, int kk){
  
  int i,j;
  dtyo fbar,mu;
  double before,after,aa,bb;
  
  before=double(time(NULL));
  
  if(_calledInterpolate==0 || kk!=_numberOfNeighbors){
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
    for(i=0;i<_dimensions;i++)_vv[i]=(vin(i)-_min[i])/(_max[i]-_min[i]);
  }
  else{
    for(i=0;i<_dimensions;i++){
      _vv[i]=vin(i);
    }
  }
  
  bb=double(time(NULL));
  _kdTreePtr->findNeighbors(_vv,_numberOfNeighbors,_neighbors,_neighborDistances);
  aa=double(time(NULL));
  
  neighborSearchTime+=aa-bb;
  
  bb=double(time(NULL));
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

  aa=double(time(NULL));
  iterationTime+=aa-bb;

  bb=double(time(NULL));
  _llt.compute(_covariance); 
  //_covarianceInverse=_covariance.inverse();
  for(i=0;i<_numberOfNeighbors;i++)_bb(i)=_function[_neighbors[i]]-fbar;
  _xx=_llt.solve(_bb);
  aa=double(time(NULL));
  
  inversionTime+=aa-bb;
  
  
  bb=double(time(NULL));
  mu=fbar;
  /*for(i=0;i<_numberOfNeighbors;i++){
    for(j=0;j<_numberOfNeighbors;j++){
      mu+=_covarianceTestPoint[i]*_covarianceInverse(i,j)*(_function[_neighbors[j]]-fbar);
    }
  }*/

  for(i=0;i<_numberOfNeighbors;i++)mu+=_covarianceTestPoint[i]*_xx(i);
  
  variance(0)=_covariogram(_vv,_vv,_dimensions,_hyperParameters)+_lambda;
  
  for(i=0;i<_numberOfNeighbors;i++)_bb(i)=_covarianceTestPoint[i];
  aa=double(time(NULL));
  iterationTime+=aa-bb;
  
  bb=double(time(NULL));
  _xx=_llt.solve(_bb);
  aa=double(time(NULL));
  varSolveTime+=aa-bb;
  
  bb=double(time(NULL));
  for(i=0;i<_numberOfNeighbors;i++){
    variance(0)-=_covarianceTestPoint[i]*_xx(i);
  } 
  aa=double(time(NULL));
  iterationTime+=aa-bb;
  
  /*for(i=0;i<_numberOfNeighbors;i++){
    variance(0)-=_covarianceTestPoint[i]*_covarianceInverse(i,i)*_covarianceTestPoint[i];
    for(j=i+1;j<_numberOfNeighbors;j++){
      variance(0)-=2.0*_covarianceTestPoint[i]*_covarianceInverse(i,j)*_covarianceTestPoint[j];
    }
  }*/
  
  variance(0)=variance(0)*_krigingParameter;
  
  _calledInterpolate=1;
  
  after=double(time(NULL));
  interpolationTime+=after-before;
  interpolationCount++;
  
  return mu;
}

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

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setLambda(dtyo ll){
  int i;
  _lambda=ll;
}

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::getCovarianceRow(int dex, ndarray::Array<dtyo,1,1> v){
  int i;
  if(_calledInterpolate==0){
    std::cout<<"You cannot call getCovarianceRow; you have not called interpolate\n";
    //printf("You can't call print gg_row; you haven't called interpolate\n");
  }
  else{
    for(i=0;i<_numberOfNeighbors;i++)v(i)=_covariance(dex,i);
  }
}

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::addPoint(ndarray::Array<dtyi,1,1> vin, dtyo f){
  int i;
  dtyi *v;
  dtyo *buff;
  
  v=new dtyi[_dimensions];
  for(i=0;i<_dimensions;i++){
    v[i]=vin(i);
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

template <typename dtyi, typename dtyo>
int GaussianProcess<dtyi,dtyo>::testKdTree(){

  return _kdTreePtr->testTree();

}

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::setHyperParameters(ndarray::Array<double,1,1> hyin){
  
  int i;
  for(i=0;i<_nHyperParameters;i++){
    _hyperParameters[i]=hyin(i);
  }
  
  
}

template <typename dtyi, typename dtyo>
void GaussianProcess<dtyi,dtyo>::getTimes(){
  std::cout<<"\n";
  //std::cout<<"interpolate time "<<interpolationTime<<"\n";
  //std::cout<<"search time "<<neighborSearchTime<<"\n";
  //std::cout<<"inversion time "<<inversionTime<<"\n";
  
  printf("interpolate time %.4e\n",interpolationTime);
  printf("search time %.4e\n",neighborSearchTime);
  printf("inversion time %.4e\n",inversionTime);
  printf("iteration time %4e\n",iterationTime);
  printf("var solve time %.4e\n",varSolveTime);
  
  std::cout<<"\n";
}

}

#define INSTANTIATEGP(dtyi,dtyo) \
	template class gptest::GaussianProcess<dtyi,dtyo>;

INSTANTIATEGP(double,double);


