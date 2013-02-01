#include "lsst/afw/math/GaussianProcess.h"
#include <iostream>

namespace lsst{
namespace afw{
namespace math{

template <typename datatype>
kd<datatype>::~kd(){
  int i;
  for(i=0;i<room;i++){
    delete [] tree[i];
  }
  delete [] tree;
}

template <typename datatype>
kd<datatype>::kd(int dd, int pp, datatype **dt, \
double(*dfn)(datatype*,datatype*,int)){
  
  int i;
  
  //buffers to use when first building the tree
  tosort=new datatype[pp];
  inn=new int[pp];
  
 
  dim=dd;
  pts=pp;
  roomstep=5000;
  room=pts;
  distance=dfn;
  data=dt;
  
  tree=new int*[pts];
  for(i=0;i<pts;i++){
    inn[i]=i;
    tree[i]=new int[4];
  }
  
  organize(inn,pts,-1,-1);
  
  delete [] tosort;
  delete [] inn;
  
}

template <typename datatype>
void kd<datatype>::organize(int *use, int ct, int parent, int dir){
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
    for(i=0;i<dim;i++){
      mean=0.0;
      var=0.0;
      for(j=0;j<ct;j++){
        mean+=data[use[j]][i];
        var+=data[use[j]][i]*data[use[j]][i];
      }
      mean=mean/double(ct);
      var=var/double(ct)-mean*mean;
      if(i==0 || var>varbest || (var==varbest && parent>=0 && i>tree[parent][0])){
        idim=i;
        varbest=var;
      }
    
    }//for(i=0;i<dim;i++)
  
    for(i=0;i<ct;i++){
      tosort[i]=data[use[i]][idim];
    }
  
    merge_sort<datatype>(tosort,use,ct);
    
    
    

    k=ct/2;
    l=ct/2;
    while(k>0 && tosort[k]==tosort[k-1])k--;
   
    while(l<ct-1 && tosort[l]==tosort[ct/2])l++;
 
    if((ct/2-k)<(l-ct/2) || l==ct-1)j=k;
    else j=l;;
    
    daughter=use[j];

    if(parent>=0)tree[parent][dir]=daughter;
    tree[daughter][0]=idim;
    tree[daughter][3]=parent;

    if(j<ct-1){
      organize(&use[j+1],ct-j-1,daughter,2);
    }
    else tree[daughter][2]=-1;
  
    if(j>0){
      organize(use,j,daughter,1);
    }
    else tree[daughter][1]=-1;
    
  }//if(ct>1)
  else{
    daughter=use[0];
    if(parent>=0)tree[parent][dir]=daughter;
    idim=tree[parent][0]+1;
    if(idim>=dim)idim=0;
    tree[daughter][0]=idim;
    tree[daughter][1]=-1;
    tree[daughter][2]=-1;
    tree[daughter][3]=parent;
    
  }
  
  if(parent==-1){
    masterparent=daughter;
  }
  
}

template <typename datatype>
void kd<datatype>::get_tree(int dex, int *v){
  v[0]=tree[dex][0];
  v[1]=tree[dex][1];
  v[2]=tree[dex][2];
  v[3]=tree[dex][3];
}

template <typename datatype>
int merge_scanner(datatype *m, int *indices, int dex, int el){
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
void merge_sort(datatype *insort, int *indices, int el){
  
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
    i=merge_scanner<datatype>(insort,indices,el/2,el);
  
    if(i>1){
      merge_sort<datatype>(insort,indices,i);
    }
    
    if(i<el-2){
      merge_sort<datatype>(&insort[i+1],&indices[i+1],el-i-1);
    }
  
  }

}

template <typename datatype>
void kd<datatype>::test_scanner(){
   
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
   
   i=merge_scanner<datatype>(mm,innt,4,5);
   
   if(i!=1){
     std::cout<<"FAILED merge_scanner i "<<i<<"\n";;
   }
   
   for(i=0;i<5;i++){
     if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
      std::cout<<"FAILED merge_scanner "<<i<<"is "<<innt[i]<<" shld "<<innshld[i]<<"\n"; 
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
   
   i=merge_scanner<datatype>(mm,innt,2,5);
   
   if(i!=0){
     std::cout<<"FAILED homog merge_scanner i "<<i<<"\n";
   }
   
   for(i=0;i<5;i++){
     if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
       std::cout<<"FAILED homog merge_scanner "<<i<<" is "<<innt[i]<<" shld "<<innshld[i]<<"\n";
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
   
   i=merge_scanner<datatype>(mm,innt,1,5);
   
   if(i!=4){
     std::cout<<"FAILED homog_b merge_scanner i "<<i<<"\n";
   }
   
   for(i=0;i<5;i++){
     if(mm[i]!=mmshld[i] || innt[i]!=innshld[i]){
       std::cout<<"FAILED homog_b merge_scanner "<<i<<" is "<<innt[i]<<" shld "<<innshld[i]<<"\n";
     }
   }
   
   delete [] mm;
   delete [] innt;
   delete [] mmshld;
   delete [] innshld;
   
}

template <typename datatype>
void kd<datatype>::test_sort(){
  
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
  
  merge_sort<datatype>(mm,innt,5);
  
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
  
  merge_sort<datatype>(mm,innt,5);
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
void kd<datatype>::walk_up(int target, int dir, int root){
  //target is the node that you are examining now
  //dir is where you came from
  //root is the ultimate point from which you started
  
  int i;
  
  if(dir==1){
    if(data[root][tree[target][0]]>=data[target][tree[target][0]]){
      std::cout<<"tree FAILURE root "<<root<<" target "<<target<<" dir "<<dir<<"\n";
      std::cout<<data[root][tree[target][0]]<<" >= "<<data[target][tree[target][0]]<<"\n";
    
    }
  }
  else{
      if(data[root][tree[target][0]]<data[target][tree[target][0]]){
      
      std::cout<<"tree FAILURE root "<<root<<"\n";
      std::cout<<" target "<<target<<" dir "<<dir<<" \n";
      std::cout<<data[root][tree[target][0]]<<" < "<<data[target][tree[target][0]]<<"\n";
      
      //printf("tree FAILURE root %d target %d dir %d %d < %d\n",
      //root,target,dir,data[root][tree[target][0]],data[target][tree[target][0]]);
    }
  }
  
  if(tree[target][3]>=0){
    if(tree[tree[target][3]][1]==target)i=1;
    else i=2;
    
    walk_up(tree[target][3],i,root);
  
  }
  
}

template <typename datatype>
void kd<datatype>::black_box_test(){

  int i,j,*isparent;
  
  /*for(i=0;i<pts;i++){
    for(j=0;j<dim;j++)printf("%d ",data[i][j]);
    printf("\n");
  }
  printf("\n\n");*/
  
  j=0;
  for(i=0;i<pts;i++){
    if(tree[i][3]<0)j++;
  }
  if(j!=1){
    std::cout<<"tree FAILURE "<<j<<" masterparents\n";
    //printf("tree FAILURE %d masterparents\n");
  }
  
  isparent=new int[pts];
  for(i=0;i<pts;i++)isparent[i]=0;
  for(i=0;i<pts;i++){
    isparent[tree[i][3]]++;
  }
  for(i=0;i<pts;i++){
    if(isparent[i]>2){ 
      std::cout<<"tree FAILURE "<<i<<" is parent to "<<isparent[i]<<"\n";
     //printf("tree FAILURE %d is parent to %d\n",i,isparent[i]);
    }
  }
  
  delete [] isparent;
  
  for(i=0;i<pts;i++){
    
    if(tree[i][3]>=0){
      if(tree[tree[i][3]][1]==i)j=1;
      else j=2;
      
     
      
      walk_up(tree[i][3],j,i);
    }
  
  }
  //printf("done with black box test\n");
  std::cout<<"done with black box test of kd_tree\n";
}

template <typename datatype>
int kd<datatype>::find_node(datatype *v){
  //this routine finds the node that would be the parent of
  //v[] if you were going to add it to the tree
  
  int consider,next,dim;
  
  dim=tree[masterparent][0];
  
  if(v[dim]<data[masterparent][dim])consider=tree[masterparent][1];
  else consider=tree[masterparent][2];
  
  next=consider;
  
  while(next>=0){
    
    consider=next;
    
    dim=tree[consider][0];
    if(v[dim]<data[consider][dim])next=tree[consider][1];
    else next=tree[consider][2];
  
  }
  
  return consider;
  
}

template<typename datatype>
void kd<datatype>::nn_srch(datatype *v, int n_nn, int *neighdex, double *dd){
  //this will search the tree for the n_nn nearest neighbors of v[];
  //the indexes of the neighbors will be stored in neighdex[];
  //the distances will be stored in dd[];
  
  int i,start;
  
  neighcand=new int[n_nn];
  ddcand=new double[n_nn];
  neighfound=0;
  neighwant=n_nn;
  
  for(i=0;i<n_nn;i++)ddcand[i]=-1.0;
  
  start=find_node(v);
  
  ddcand[0]=distance(data[start],v,dim);
  neighcand[0]=start;
  neighfound=1;
  
  if(tree[start][3]>=0){
    nn_explore(v,tree[start][3],start);
  }
  if(tree[start][1]>=0){
    nn_explore(v,tree[start][1],start);
  }
  if(tree[start][2]>=0){
    nn_explore(v,tree[start][2],start);
  }
  
  for(i=0;i<n_nn;i++){
    neighdex[i]=neighcand[i];
    dd[i]=ddcand[i];
  }
  
  delete [] neighcand;
  delete [] ddcand;
  
}

template<typename datatype>
void kd<datatype>::nn_explore(datatype *v, int consider, int from){
  //in the process of searching for nearest neighbors of v[]
  //this routine will look at the point denoted by 'consider'
  //and walk up/down the tree based on where it is coming from ('from')
  
  int i,j,going;
  double dd;
  
  
  
  dd=distance(data[consider],v,dim);
  
  /*for(i=1;i<neighfound;i++){
    if(ddcand[i]<ddcand[i-1]){
      printf("NEIGHFOUND out of order\n");
      exit(1);
    }
  }*/
  
  if(neighfound<neighwant || dd<ddcand[neighwant-1]){
    for(i=0;i<neighfound && ddcand[i]<dd;i++);
    j=i;
  
      
    for(i=neighwant-1;i>j;i--){
      ddcand[i]=ddcand[i-1];
      neighcand[i]=neighcand[i-1];
    }
    
    ddcand[j]=dd;
    neighcand[j]=consider;
    
    if(neighfound<neighwant)neighfound++;
  }
  
  if(tree[consider][3]==from){
    //you came here from the parent
    
    i=tree[consider][0];
    dd=v[i]-data[consider][i];
    if((dd<=ddcand[neighfound-1] || neighfound<neighwant) \
    && tree[consider][1]>=0){
      nn_explore(v,tree[consider][1],consider);
    }
    
    dd=data[consider][i]-v[i];
    if((dd<=ddcand[neighfound-1] || neighfound<neighwant) \
    && tree[consider][2]>=0){
      nn_explore(v,tree[consider][2],consider);
    }
  }
  else{
    //you came here from one of the branches
    
    //descend the other branch
    if(tree[consider][1]==from){
      going=2;
    }
    else{ 
      going=1;
    }
    
    j=tree[consider][going];
    
    if(j>=0){
      i=tree[consider][0];
      if(going==1)dd=v[i]-data[consider][i];
      else dd=data[consider][i]-v[i];
      
      if(dd<=ddcand[neighfound-1] || neighfound<neighwant){
        nn_explore(v,j,consider);
      }
    }
    
    //ascend to the parent
    if(tree[consider][3]>=0){
      
      nn_explore(v,tree[consider][3],consider);
      
    }
    
  }
 

}



#define INSTANTIATE_kd(dty) \
	template kd<dty>::kd(int,int,dty**,double (*)(dty*,dty*,int)); \
	template void merge_sort<dty>(dty*,int*,int); \
	template int merge_scanner<dty>(dty*,int*,int,int);\
	template void kd<dty>::test_scanner(); \
	template void kd<dty>::test_sort(); \
	template void kd<dty>::get_tree(int,int*);\
	template void kd<dty>::black_box_test();\
	template kd<dty>::~kd();\
	template void kd<dty>::nn_srch(dty*,int,int*,double*);
	

INSTANTIATE_kd(double)
INSTANTIATE_kd(int)

template <typename dtyi, typename dtyo>
gaussianprocess<dtyi,dtyo>::~gaussianprocess(){
  int i;
  
  if(maxmin==1){
    for(i=0;i<room;i++){
      delete [] data[i];
    }
    delete [] data;
    
    if(called_interp==1)delete [] vv;
    
  }
  
  delete kptr;
  
  if(called_interp==1){
    delete [] neigh;
    delete [] ddneigh;
    delete [] ggq;
    gg.resize(0,0);
    
  }
  
}

template <typename dtyi, typename dtyo>
gaussianprocess<dtyi,dtyo>::gaussianprocess(int dd, int pp, dtyi **datain, dtyi *mx, dtyi *mn, dtyo *ff,\
double(*dfn)(dtyi*,dtyi*,int), dtyo(*cfn)(dtyi*,dtyi*,int)){
 //constructor if you have maxs and mins
  
  int i,j;
  
  dim=dd;
  pts=pp;
  room=pts;
  
  fn=ff;
  
  kriging_parameter=dtyo(1.0);
  
  etime=0.0;
  ltime=0.0;
  ect=0.0;
  lct=0.0;
  
  max=mx;
  min=mn;
  
  covariogram=cfn;
  distance=dfn;
  
  called_interp=0;

  lambda=new dtyo[pts];
  for(i=0;i<pts;i++){
    lambda[i]=dtyo(1.0e-5);
  }
  
  maxmin=1;
  data=new dtyi*[pts];
  for(i=0;i<pts;i++){
    data[i]=new dtyi[dim];
    for(j=0;j<dim;j++){
      data[i][j]=(datain[i][j]-min[j])/(max[j]-min[j]);
    }
  }
   
  kptr=new kd<dtyi>(dim,pts,data,distance);
  
}

template <typename dtyi, typename dtyo>
gaussianprocess<dtyi,dtyo>::gaussianprocess(int dd, int pp, dtyi **datain, dtyo *ff,\
double(*dfn)(dtyi*,dtyi*,int), dtyo(*cfn)(dtyi*,dtyi*,int)){
 //constructor if you do not have maxs and mins
  
  int i;
  
  dim=dd;
  pts=pp;
  room=pts;
  
  fn=ff;
  kriging_parameter=dtyo(1.0);
  
  covariogram=cfn;
  distance=dfn;
  
  called_interp=0;
  
  lambda=new dtyo[pts];
  for(i=0;i<pts;i++){
    lambda[i]=dtyo(1.0e-5);
  }
  maxmin=0;
  data=datain;
   
  kptr=new kd<dtyi>(dim,pts,data,distance);
  
  etime=0.0;
  ltime=0.0;
  ect=0.0;
  lct=0.0;
  
}

template <typename dtyi, typename dtyo>
void gaussianprocess<dtyi,dtyo>::set_kp(int kk){
  
  Eigen::Matrix <dtyo,Eigen::Dynamic,Eigen::Dynamic> kgg,kggin;
  
  int *kneigh,i,j,k,*inn;
  double *ddneigh;
  dtyo *kggq,mu,sig2,fbar,*rat;

  kneigh=new int[kk+1];
  ddneigh=new double[kk+1];
  kggq=new dtyo[kk];
  
  inn=new int[pts];
  rat=new dtyo[pts];
  
  kgg.resize(kk,kk);
  
  for(i=0;i<pts;i++){

    kptr->nn_srch(data[i],kk+1,kneigh,ddneigh);
    
    for(j=0;j<kk;j++){
      kggq[j]=covariogram(data[i],data[kneigh[j+1]],dim);
      
      kgg(j,j)=covariogram(data[kneigh[j+1]],data[kneigh[j+1]],dim)\
      +lambda[kneigh[j+1]];
      
      for(k=j+1;k<kk;k++){
        kgg(j,k)=covariogram(data[kneigh[j+1]],data[kneigh[k+1]],dim);
        kgg(k,j)=kgg(j,k);
      }
    }
     
    kggin=kgg.inverse();
    
    fbar=dtyo(0.0);
    for(j=0;j<kk;j++){
      fbar+=fn[kneigh[j+1]];
    }
    fbar=fbar/dtyo(kk);
    
    mu=fbar;
    for(j=0;j<kk;j++){
      for(k=0;k<kk;k++){
        mu+=kggq[j]*kggin(j,k)*(fn[kneigh[k+1]]-fbar);
      }
    }
    
    sig2=covariogram(data[i],data[i],dim)+lambda[kneigh[1]];
    
    for(j=0;j<kk;j++){
      sig2-=kggq[j]*kggin(j,j)*kggq[j];
      for(k=j+1;k<kk;k++){
         sig2-=2.0*kggq[j]*kggin(j,k)*kggq[k];
      }
    }
    
    rat[i]=(mu-fn[i])*(mu-fn[i])/sig2; 
    
  }
  
  merge_sort<dtyo>(rat,inn,pts);
  
  kriging_parameter=rat[68*pts/100];
  
  delete [] kneigh;
  delete [] ddneigh;
  delete [] kggq;
  delete [] inn;
  delete [] rat;
  
  
}

template <typename dtyi, typename dtyo>
dtyo gaussianprocess<dtyi,dtyo>::interpolate(dtyi *vin, dtyo *sig2, int kk){
  
  int i,j;
  dtyo fbar,mu;
  double before,after;
  
  
  if(called_interp==0 || kk!=n_nn){
     if(called_interp==1){
       delete [] ggq;
       delete [] neigh;
       delete [] ddneigh;
       
     }
     
     ggq=new dtyo[kk];
     gg.resize(kk,kk);
     
     neigh=new int[kk];
     ddneigh=new double[kk];
     
     n_nn=kk;
  }
  
  if(maxmin==1 && called_interp==0){
    vv=new dtyi[dim];
  }
  
  if(maxmin==1){
    for(i=0;i<dim;i++)vv[i]=(vin[i]-min[i])/(max[i]-min[i]);
  }
  else vv=vin;
  
  
  kptr->nn_srch(vv,n_nn,neigh,ddneigh);
  
  fbar=0.0;
  for(i=0;i<n_nn;i++)fbar+=fn[neigh[i]];
  fbar=fbar/double(n_nn);
  
  
  
  
  for(i=0;i<n_nn;i++){
    ggq[i]=covariogram(vv,data[neigh[i]],dim);
    gg(i,i)=covariogram(data[neigh[i]],data[neigh[i]],dim)+lambda[neigh[i]];
    for(j=i+1;j<n_nn;j++){
      gg(i,j)=covariogram(data[neigh[i]],data[neigh[j]],dim);
      gg(j,i)=gg(i,j);
    }
  }
  
  before=double(time(NULL));
  ggin=gg.inverse();
  after=double(time(NULL));
  etime+=after-before;
  ect+=1.0;
  
  mu=fbar;
  for(i=0;i<n_nn;i++){
    for(j=0;j<n_nn;j++){
      mu+=ggq[i]*ggin(i,j)*(fn[neigh[j]]-fbar);
    }
  }
  
  sig2[0]=covariogram(vv,vv,dim)+lambda[neigh[0]];
  
  for(i=0;i<n_nn;i++){
    sig2[0]-=ggq[i]*ggin(i,i)*ggq[i];
    for(j=i+1;j<n_nn;j++){
      sig2[0]-=2.0*ggq[i]*ggin(i,j)*ggq[j];
    }
  }
  
  sig2[0]=sig2[0]*kriging_parameter;
  
  called_interp=1;
  

  
  return mu;
}

template <typename dtyi, typename dtyo>
void gaussianprocess<dtyi,dtyo>::print_nn(int *v){
  int i;
  if(called_interp==0){
    std::cout<<"You cannot call print_nn; you have not called interpolate at all\n";
    //printf("You cannot call print_nn; you haven't called interpolate at all\n");
  }
  else{
    for(i=0;i<n_nn;i++)v[i]=neigh[i];
  }
}

template <typename dtyi, typename dtyo>
void gaussianprocess<dtyi,dtyo>::set_lambda(dtyo ll){
  int i;
  for(i=0;i<pts;i++){
   lambda[i]=ll;
  }
}

template <typename dtyi, typename dtyo>
void gaussianprocess<dtyi,dtyo>::print_ggrow(int dex, dtyo *v){
  int i;
  if(called_interp==0){
    std::cout<<"You cannot call print gg_row; you have not called interpolate\n";
    //printf("You can't call print gg_row; you haven't called interpolate\n");
  }
  else{
    for(i=0;i<n_nn;i++)v[i]=gg(dex,i);
  }
}

/*#define INSTANTIATE_gaussianprocess(dtyi,dtyo) \
	template gaussianprocess<dtyi,dtyo>::gaussianprocess(int,int,dtyi**,dtyo*,\
	double(*)(dtyi*,dtyi*,int),dtyo(*)(dtyi*,dtyi*,int));\
        template gaussianprocess<dtyi,dtyo>::gaussianprocess(int,int,dtyi**,dtyi*,dtyi*,dtyo*,\
	double(*)(dtyi*,dtyi*,int),dtyo(*)(dtyi*,dtyi*,int)); \
	template dtyo gaussianprocess<dtyi,dtyo>::interpolate(dtyi*,dtyo*,int);\
	template void gaussianprocess<dtyi,dtyo>::print_nn(int*);\
	template void gaussianprocess<dtyi,dtyo>::set_lambda(dtyo);\
	template void gaussianprocess<dtyi,dtyo>::print_ggrow(int,dtyo*);\
	template void gaussianprocess<dtyi,dtyo>::set_kp(int);

INSTANTIATE_gaussianprocess(double,double);*/

}}}
