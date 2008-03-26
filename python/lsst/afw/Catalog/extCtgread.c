/*** File libwcs/ctgread.c
 *** January 18, 2005
 *** By Doug Mink, dmink@cfa.harvard.edu
 *** Harvard-Smithsonian Center for Astrophysics
 *** Copyright (C) 1998-2005
 *** Smithsonian Astrophysical Observatory, Cambridge, MA, USA

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
    
    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Correspondence concerning WCSTools should be addressed as follows:
           Internet email: dmink@cfa.harvard.edu
           Postal address: Doug Mink
                           Smithsonian Astrophysical Observatory
                           60 Garden St.
                           Cambridge, MA 02138 USA
 */

/* int ctgread()	Read catalog stars in specified region of the sky
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

/* Definitions for Python <-> C interface */
#include "Python.h"
#include "numarray/libnumarray.h"
#include "numarray/arrayobject.h"
#include "libwcs/wcscat.h"

#define MAX_MAG_INDEX 5

/* if DEBUG defined, then developmental diagnostics printed */
/*#define DEBUG 1*/


/* CTGREAD -- Read ASCII stars in specified region */

/*
int
ctgread (catfile, refcat, distsort, cra, cdec, dra, ddec, drad, dradi,
	 sysout, eqout, epout, mag1, mag2, sortmag, nsmax, starcat,
	 o_tnum, o_tra, o_tdec, o_tpra, o_tpdec, tmag, o_tc, tobj, nlog)

char	*catfile;	
int	refcat;		
int	distsort;	
double	cra;		
double	cdec;		
double	dra;		
double	ddec;		
double	drad;		
double	dradi;		
int	sysout;		
double	eqout;		
double	epout;		
double	mag1,mag2;	
int	sortmag;	
int	nsmax;		
struct StarCat **starcat; 
double	*o_tnum;		
double	*o_tra;		
double	*o_tdec;		
double	*o_tpra;		
double	*o_tpdec;		
double	**tmag;		
int	*o_tc;	
char	**tobj;		
int	nlog;           
*/

/*========================================================================== */
                                                                                
/* Python extension wrapper for ctgread */
static PyObject * Py_ctgread(PyObject *obj, PyObject *args) {

    /* define the C scalar input arguments*/
    int i_refcat, i_distsort, i_sysout, i_sortmag, i_nsmax, i_nlog;
    char *i_catfile;
    double i_cra, i_cdec, i_dra, i_ddec, i_drad, i_dradi, i_eqout, i_epout,
        i_mag1, i_mag2;

    /* define PyArrayObjects for all array arguments */
    PyArrayObject *o_tnum, *o_tra, *o_tdec, *o_tpra, *o_tpdec, *o_tc, *o_tmag;

    /* define place holders for unused parameters: tobj and starcat */
    char *t_tobj;
    struct StarCat **ptr_ptr_starcat, *ptr_starcat;
    int ret_star, t_count, dimsize, i; 
    double **tmag;

/*--------------------------------------------------------------------- */
/* N O T E    N O T E     N O T E     N O T E     N O T E     N O T E   */
/* python ctgread interface does not return tobj; 
   'c' ctgread implemented: if tobj==NULL, tobj not be used. */
/* N O T E    N O T E     N O T E     N O T E     N O T E     N O T E   */
/*--------------------------------------------------------------------- */
    ptr_ptr_starcat = &ptr_starcat;
    ptr_starcat = NULL;
    t_tobj = NULL;


#ifdef DEBUG
    printf("Py_ctgread: Just Arrived in wrapper\n");
#endif

    /* Parse input arguments to the Python wrapper routine for ctgread */
    if (!PyArg_ParseTuple(args,"siiddddddiddddiii",
        &i_catfile,
        &i_refcat,
        &i_distsort,
        &i_cra,
        &i_cdec,
        &i_dra,
        &i_ddec,
        &i_drad,
        &i_dradi,
        &i_sysout,
        &i_eqout,
        &i_epout,
        &i_mag1,
        &i_mag2,
        &i_sortmag,
        &i_nsmax,
        &i_nlog)) {
        PyErr_SetString(PyExc_TypeError,"ctgread: Invalid parameters.");
        goto _failbeg;
    }

    /* for output arrays:
        build dimensioned numarrays to pass (appropriately) to 'C'
        These arrays need to be PY_DECREF when done with */
    dimsize = (i_nsmax < 1) ? 1:i_nsmax;
    int dimensions[2];
    dimensions[0] = MAX_MAG_INDEX;
    dimensions[1] = dimsize;
    o_tnum = (PyArrayObject *)PyArray_FromDims(1,&dimsize,PyArray_DOUBLE);
    o_tra  = (PyArrayObject *)PyArray_FromDims(1,&dimsize,PyArray_DOUBLE);
    o_tdec = (PyArrayObject *)PyArray_FromDims(1,&dimsize,PyArray_DOUBLE);
    o_tpra = (PyArrayObject *)PyArray_FromDims(1,&dimsize,PyArray_DOUBLE);
    o_tpdec= (PyArrayObject *)PyArray_FromDims(1,&dimsize,PyArray_DOUBLE);
    o_tc   = (PyArrayObject *)PyArray_FromDims(1,&dimsize,PyArray_INT);
    o_tmag = (PyArrayObject *)PyArray_FromDims(2,dimensions,PyArray_DOUBLE);
    if (!o_tnum || !o_tra || !o_tdec || !o_tpra || 
        !o_tpdec || !o_tc  || !o_tmag ) {
        PyErr_SetString(PyExc_TypeError,"ctgread: Couldn't create Output 1-D Arrays.");
        goto _fail;
    }

  /* Build a 'C' style 2 D array from the Python array object */
  if ( !(tmag= (double **)malloc(dimensions[0] * sizeof(double*)))) goto _fail;

  int j;
  /* load the pointers to the vectors */
  for (i = 0; i < dimensions[0]; i++) {
      tmag[i] = (double *)(o_tmag->data + i * o_tmag->strides[0]);

      /* initialize the future mag values to null - as per scat */
      for (j = 0; j < dimensions[1]; j++) 
          tmag[i][j] = 99.0;
  }

#ifdef DEBUG
    printf("Created output arrays\n");
    printf("Just before 'C' ctgread\n");
#endif

    /* invoke the wrapped C library routine */
//    t_count = ctgread(
//        (char*)i_catfile, 
//        (int)i_refcat, 
//        (int)i_distsort, 
//        (double)i_cra, 
//        (double)i_cdec, 
//        (double)i_dra, 
//        (double)i_ddec, 
//        (double)i_drad, 
//        (double)i_dradi,
//        (int)i_sysout, 
//        (double)i_eqout, 
//        (double)i_epout, 
//        (double)i_mag1, 
//        (double)i_mag2, 
//        (int)i_sortmag, 
//        (int)i_nsmax, 
//        (struct StarCat **)ptr_ptr_starcat,
//        (double *)NA_OFFSETDATA(o_tnum), 
//        (double *)NA_OFFSETDATA(o_tra), 
//        (double *)NA_OFFSETDATA(o_tdec), 
//        (double *)NA_OFFSETDATA(o_tpra), 
//        (double *)NA_OFFSETDATA(o_tpdec), 
//        (double**)tmag,
//        (double *)NA_OFFSETDATA(o_tc), 
//        (char **)&t_tobj,
//        (int)i_nlog );
    t_count = ctgread(
        i_catfile, 
        i_refcat, 
        i_distsort, 
        i_cra, 
        i_cdec, 
        i_dra, 
        i_ddec, 
        i_drad, 
        i_dradi,
        i_sysout, 
        i_eqout, 
        i_epout, 
        i_mag1, 
        i_mag2, 
        i_sortmag, 
        i_nsmax, 
        (struct StarCat **)ptr_ptr_starcat,
        NA_OFFSETDATA(o_tnum), 
        NA_OFFSETDATA(o_tra), 
        NA_OFFSETDATA(o_tdec), 
        NA_OFFSETDATA(o_tpra), 
        NA_OFFSETDATA(o_tpdec), 
        tmag,
        NA_OFFSETDATA(o_tc), 
        &t_tobj,
        i_nlog );

#ifdef DEBUG
    printf("Just returned from actual 'C' ctgread\n");
    printf("dimsize: %d  t_count: %d i_nsmax %d \n",dimsize,t_count,i_nsmax);
#endif
        ret_star = (t_count < dimsize)? t_count:dimsize;
#ifdef DEBUG
        printf("Returning with %d stars out of %d available\n",ret_star,t_count);
#endif

    /* Check if valid results  for output 1-D arrays */
    if (( o_tnum == Py_None) || ( o_tra == Py_None) ||
        ( o_tdec == Py_None) || ( o_tpra == Py_None) ||
        ( o_tpdec == Py_None) || ( o_tc == Py_None) ){
        PyErr_SetString(PyExc_TypeError,"ctgread: output 1-D numeric arrays are invalid.");
        goto _fail;
    }
#ifdef DEBUG
        printf("Output numeric arrays check ok \n");
#endif

    /* Return all results */
#ifdef DEBUG
    printf("Return star count: %d.\n",ret_star);
    for ( i = 0; i < ret_star; i++) {
        printf("i:%d tmag[0][i]:%g tmag[1][i]:%g  tmag[2][i]:%g tmag[3][i]:%g\n", i,tmag[0][i],tmag[1][i],tmag[2][i],tmag[3][i]);
    }
    printf("Building python return list and returning%d\n",ret_star);
#endif
    if ( ret_star) {
        return Py_BuildValue("iOOOOOOO",ret_star,o_tnum,o_tra,o_tdec,o_tpra,
               o_tpdec,o_tmag,o_tc);
    }

    /* drop thru to fail if count indicates ctgread failure */
#ifdef DEBUG
    else printf("Failed to match catalog stars\n");
#endif

_fail:
#ifdef DEBUG
    printf("_fail\n");
#endif
    /* Decrement use count for output array which won't be used in Python  */
    Py_XDECREF(o_tnum);
    Py_XDECREF(o_tra);
    Py_XDECREF(o_tdec);
    Py_XDECREF(o_tpra);
    Py_XDECREF(o_tpdec);
    Py_XDECREF(o_tc);
    Py_XDECREF(o_tmag); 

_failbeg:
#ifdef DEBUG
    printf("_failbeg\n");
#endif
    return NULL;
}
                                                                                
/*=======================================================================*/

PyDoc_STRVAR(module_doc, "See original wcstools:ctgread.c for interface specification.");


/*=======================================================================*/


static PyMethodDef ctgreadMethods[] = {
{"ctgread", Py_ctgread, METH_VARARGS,
"retrieve data from specified astronomical catalog"},
{NULL,NULL,0,NULL} /* Sentinel */
};
                                                                                
PyMODINIT_FUNC initctgread(void) {
    (void)Py_InitModule3("ctgread",ctgreadMethods,module_doc);
    import_libnumarray();
    import_array();
}
                                                                                
/*=======================================================================*/


