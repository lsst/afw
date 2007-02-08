/* File:  	ctgread.i 						*/
/* Author:	Michelle Miller						*/
/* Date: 	10/28/05						*/
/*                                                                      */
/* Wrap wcstools function ctgread for extracting a portion of a catalog */
/* to be used in image matching to determine wcs.                       */

%define ctgread_DOCSTRING
"
Basic wrappers for wcslib's ctgread
"
%enddef

%feature("autodoc", "1");
%module(docstring=ctgread_DOCSTRING) ctgread

%{
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include "wcs.h"
#include "wcscat.h"
#include "fitsfile.h"
%}

extern char catdir[64]="/data/catalogs";

#define MAX_LTOK	80

%include <carrays.i>
%array_functions (double, doubleArray);
%array_functions (int, intArray);

%include <typemaps.i>

/* malloc space in python before call ctgread(), then argout typemap copies */
/* each C data element in array to same place in the python array malloced  */
/* before call                                                              */
%typemap(in) double *OUTPUT {
   int i;
   if (!PySequence_Check ($input)) {
      PyErr_SetString (PyExc_ValueError, "Expected a sequence");
      return NULL;
   }
   
   if (PySequence_Length ($input) != $1_dim0) {
      PyErr_SetString (PyExc_ValueError, "Size mismatch. Expected %d elements", $1_dim0);
      return NULL;
   }

   $1 = (double *)malloc ($1_dim0*sizeof(double));
   for (i=0; i < $1_dim0; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyNumber_Check(o)) {
         $1[i] = (double) PyFloat_AsDouble(o); /* right call? type? */
      } else {
         PyErr_SetString (PyExc_ValueError, "Sequence of elements must be numbers");
         free($1);
         return NULL;
      }
   }
}

%typemap(freearg) double *OUTPUT {
   if ($1) free($1);
}

%typemap(argout) double *OUTPUT {
/* create ptr and place in output */
   if ($1) {
      PyObject *o = PyFloat_FromDouble(*$1);
      $result = t_output_helper($result,o);   /* append to output */
   } 
}

%typemap(in) double **OUTPUT {
   int i,j;
   if (!PySequence_Check ($input)) {
      PyErr_SetString (PyExc_ValueError, "Expected a sequence");
      return NULL;
   }
   
   if (PySequence_Length ($input) != ($1_dim0*$1_dim1)) {
      PyErr_SetString (PyExc_ValueError, "Size mismatch. Expected %d elements", $1_dim0);
      return NULL;
   }

   $1 = (double **)malloc ($1_dim0*sizeof(double *));
   for (i=0; i < $1_dim0; i++) {
      $1[i] = (double *)malloc ($1_dim1*sizeof(double));
   }
   for (i=0; i < $1_dim0; i++) {
      for (j=0; j < $1_dim1; j++) {
         PyObject *o = PySequence_GetItem($input,i[j]);
         if (PyNumber_Check(o)) {
            $1[i][j] = (double) PyFloat_AsDouble(o); /* right call? type? */
         } else {
            PyErr_SetString (PyExc_ValueError, "Sequence of elements must be numbers");
            free($1);
            return NULL;
         }
      }
   }
}

%typemap(freearg) double **OUTPUT {
   if ($1) free($1);
}

%typemap(argout) double **OUTPUT {
/* create ptr and place in output */
   if (*$1) {
      PyObject *o = PyFloat_FromDouble(**$1);
      $result = t_output_helper($result,o);   /* append to output */
   } 
}

%typemap(in) char **OUTPUT {
   int i,j;
   if (!PySequence_Check ($input)) {
      PyErr_SetString (PyExc_ValueError, "Expected a sequence");
      return NULL;
   }
   
   if (PySequence_Length ($input) != ($1_dim0*$1_dim1)) {
      PyErr_SetString (PyExc_ValueError, "Size mismatch. Expected %d elements", $1_dim0);
      return NULL;
   }

   $1 = (char **)malloc ($1_dim0*sizeof(char *));
   for (i=0; i < $1_dim0; i++) {
      $1[i] = (char *)malloc ($1_dim1*sizeof(char));
   }
   for (i=0; i < $1_dim0; i++) {
         PyObject *o = PySequence_GetItem($input,i);
         if (PyString_Check(o)) 
            $1[i] = PyString_AsString(o); 
         else {
            PyErr_SetString (PyExc_ValueError, "Sequence of elements must be strings");
            free($1);
            return NULL;
         }
   }
}

%typemap(freearg) char **OUTPUT {
   if ($1) free($1);
}

%typemap(argout) char **OUTPUT {
/* create ptr and place in output */
   if (*$1) {
      PyObject o = PyString_FromString(*$1);
      $result = t_output_helper($result,o);   /* append to output */
   } 
}

%typemap(in) int *OUTPUT {
   int i;
   if (!PySequence_Check ($input)) {
      PyErr_SetString (PyExc_ValueError, "Expected a sequence");
      return NULL;
   }
   
   if (PySequence_Length ($input) != $1_dim0) {
      PyErr_SetString (PyExc_ValueError, "Size mismatch. Expected %d elements", $1_dim0);
      return NULL;
   }

   $1 = (int *)malloc ($1_dim0*sizeof(int));
   for (i=0; i < $1_dim0; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyNumber_Check(o)) {
         $1[i] = (int) PyInt_AsLong(o); 
      } else {
         PyErr_SetString (PyExc_ValueError, "Sequence of elements must be numbers");
         free($1);
         return NULL;
      }
   }
}

%typemap(freearg) int *OUTPUT {
   if ($1) free($1);
}

%typemap(argout) int *OUTPUT {
/* create ptr and place in output */
   if ($1) {
      PyObject *o = PyInt_FromLong(*$1);
      $result = t_output_helper($result,o);   /* append to output */
   } 
}

/* C helper functions */
%inline %{
struct StarCat **new_ptr_to_ptr_StarCat (struct StarCat *sp) {
   struct StarCat **spp;
   spp = &sp;
   return (spp);
}

/* make this setup match scat.c */
struct StarCat **createNullStarCat (int array_size) 
{
   struct StarCat **spp;
   int i;

   spp = (struct StarCat **)malloc (array_size*sizeof(struct StarCat *));
   for (i=0; i < array_size; i++) {
      spp[i] = NULL;
   }
   return (spp);
}

double **new2D_DoubleArray (int rows, int cols, double init_value) 
{
   int i,j;
   double **new2D;

   new2D = (double **)malloc(rows*sizeof(double *));
   for (i=0; i < rows; i++) {
      new2D[i] = (double *)malloc(cols*sizeof(double));
   }
   for (i=0; i < rows; i++)
      for (j=0; j < cols; j++)
         *(*(new2D+i)+j) = init_value;
   return(new2D);
}

char **new2D_charArray (int rows, int cols, char *init_value)
{
   int i;
   char **new2D = (char **)malloc(rows*sizeof(char *));

   for (i=0; i < rows; i++) {
      new2D[i] = (char *)malloc (cols*sizeof(char));
   }
   for (i = 0; i < rows; i++)
      strcpy (new2D[i], init_value);
   return(new2D);
}

char **nullStringArray() {
   return (NULL);
}
%}

extern int ctgread (char *catfile, int refcat, int distsort, double cra,
                    double cdec, double dra, double ddec, double drad,
                    double dradi, int sysout, double eqout, double epout,
                    double mag1, double mag2, int sortmag, int nsmax, 
                    struct StarCat **starcat, double *tnum, double *tra,
                    double *tdec, double *tpra, double *tpdec, double **tmag,
                    int *tc, char **tobj, int nlog);

extern struct StarCat * ty2open (int nstar, int nread);

