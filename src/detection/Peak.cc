/*****************************************************************************/
/**
 * \file
 *
 * \brief Handle Peak%s
 */
#include <cassert>
#include <string>
#include <typeinfo>
#include <boost/format.hpp>
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/Footprint.h"

namespace lsst {
namespace afw {
namespace detection {

int Peak::id = 0;                       //!< Counter for Peak IDs

/**
 * Return a string-representation of a Peak
 */
std::string Peak::toString() {
    return (boost::format("%d: (%d,%d)  (%.3f, %.3f)") % _id % _ix % _iy % _fx % _fy).str();
}

/************************************************************************************************************/
/*
 * Find all peaks in an object, specified as an Footprint and Image, sorted in
 * decreasing order of peak intensity
 *
 * Return a boost::shared_ptr<PeakList> of the peaks found
 */
template<typename ImageT>
boost::shared_ptr<Peak::List> peaksFindInFootprint(ImageT const& reg, ///< region where object was found
                                            Footprint const& foot,    ///< set of of pixels in object
                                            int const delta,          ///< amount peak must exceed neighbours
                                            int const npeak_max       ///< max. number of peaks to find; unlimited if < 0
                                           ) {
    
    boost::shared_ptr<Peak::List> peaks(new Peak::List);
#if 0
   int a;				// a resolved plateau alias */
   int cmin;				// == om->cmin, unpacked for compiler*/
   int i,j,k;
   int nspan;				// unpack obj->nspan for compiler */
   PIX *rm2, *rm1, *r0, *rp1;		// rows [y-2], [y-1], [y], and [y+1] */
   int *idm1, *id0;			// plateau IDs in previous and this row
					   (m1 stands for "-1": ID[y-1]) */
   int is_peak;				// is this pixel a peak? */
   int next_id;				// next available plateau id */
   PIX max = 0;				// globally maximum pixel in om */
   int max_x = 0, max_y = 0;		// position of maximum */
   int mwidth;				// mask width allowing for margin */
   PIX *null;				// line of 0s */
   int plateau_size;			// size of plateau[] */
   PLATEAU *plateau;			// book keeping for plateaus */
   PIX **rows;				// unpacked from reg->ROWS */
   int row0, col0;			// unpacked from reg->{row,col}0 */
   int nrow, ncol;			// unpacked from reg->n{row,col} */
   SPAN *spans;				// SPANs in this OBJECT */
   PEAK *tmpeak;			// used in PEAK shuffling */
   int y, x1, x2;			// unpacked from a SPAN */
   int *tmp;				// used in switching idm1 and id0 */
   PIX v00;				// == r0[j] */

   shAssert(peaks != NULL);
   shAssert(delta > 0);
   shAssert(om != NULL && om->nspan > 0);
   shAssert(reg != NULL && reg->type == TYPE_PIX);
#if 0					// apparently not needed; PR 6246 */
   shAssert(reg->ncol >= 3);
#endif

   spans = om->s;
   nspan = om->nspan;
   cmin = om->cmin;
   rows = reg->ROWS;
   row0 = reg->row0; col0 = reg->col0;
   nrow = reg->nrow; ncol = reg->ncol;

   shAssert(om->rmin >= row0 && om->rmax < row0 + nrow);
   shAssert(om->cmin >= col0 && om->cmax < col0 + ncol);

   if(peaks->size == 0) {
      phPeaksRenew(peaks,1);
   }
/*
 * We need id arrays that are as wide as the object, with an extra
 * column on each side. We'll adjust the pointers so that the indexing
 * in these arrays is the same as in the data itself
 */
   mwidth = 1 + (om->cmax - om->cmin + 1) + 1; // mask width + margin */

   idm1 = alloca(2*mwidth*sizeof(int));
   id0  = idm1 + mwidth;
   null = alloca(mwidth*sizeof(PIX));
   memset(id0,'\0',mwidth*sizeof(int));
   memset(null,'\0',mwidth*sizeof(PIX));
   shAssert(id0[0] == 0 && null[0] == 0); // check that 0 is all bits 0 */

   idm1 -= cmin - col0 - 1;		// match indexing to rows */
   id0 -= cmin - col0 - 1;
   null -= cmin - col0 - 1;

   next_id = 0;
   plateau_size = (om->npix > 25600) ? 25600 : om->npix;
   plateau = alloca(plateau_size*sizeof(PLATEAU));
      
   y = spans[0].y - row0;

   rm1 = r0 = null;			// all zeros */
   rp1 = rows[y];
/*
 * time to start processing the image. The task of finding all maxima
 * is significantly complicated by the need to deal with plateaus where
 * all pixel values are the same.
 *
 * I deal with the problem by
 *  1/ finding all absolute maxima
 *  2/ finding all plateaux (by a technique similar to the usual object
 *     finder)
 *  3/ note which plateaux are adjacent to a higher pixel; all others
 *     are peaks in their own right.
 */
   for(i = 0;i <= nspan;) {
      if(npeak_max >= 0 && peaks->npeak >= npeak_max) {
	 break;
      }

      tmp = idm1; idm1 = id0; id0 = tmp; // switch idm1 and id0 */
      memset(&id0[cmin - col0 - 1],'\0',mwidth*sizeof(int));
      rm2 = rm1;
      rm1 = r0;
      r0 = rp1;

      if(i == nspan) {			// analyse last line in mask */
	 y++; i++;
	 rp1 = null;			// all zeros */
      } else {
	 x1 = spans[i].x1 - col0;
	 x2 = spans[i].x2 - col0;
	 y = spans[i].y - row0;
	 rp1 = (y + 1 >= nrow) ? null : rows[y + 1];

	 do {
	    x1 = spans[i].x1 - col0; x2 = spans[i].x2 - col0;
	    
	    for(j = x1;j <= x2;j++) {
/*
 * look for maxima
 */
	       v00 = r0[j];

	       if(v00 > max) {		// look for global maximum,
					   ignoring value of delta */
		  max = v00;
		  max_x = j;
		  max_y = y;
	       }

	       is_peak =
		 (v00 - rm1[j] >= delta && v00 - rp1[j] >= delta) ? 1 : 0;
	       if(is_peak && j > 0) {
		  is_peak = (v00 - rm1[j-1] >= delta &&
			     v00 -  r0[j-1] >= delta &&
			     v00 - rp1[j-1] >= delta) ? 1 : 0;
	       }
	       if(is_peak && j < ncol - 1) {
		  is_peak = (v00 - rm1[j+1] >= delta &&
			     v00 -  r0[j+1] >= delta &&
			     v00 - rp1[j+1] >= delta) ? 1 : 0;
	       }
	       
	       if(is_peak) {
		  if(peaks->npeak >= peaks->size) {
		     phPeaksRealloc(peaks, 2*peaks->size);
		  }
		  peaks->peaks[peaks->npeak]->peak = v00;
		  peaks->peaks[peaks->npeak]->rpeak = y;
		  peaks->peaks[peaks->npeak]->cpeak = j;
		  peaks->peaks[peaks->npeak]->rowc = y + 0.4999;
		  peaks->peaks[peaks->npeak]->colc = j + 0.4999;
		  peaks->npeak++;

		  continue;
	       }
/*
 * now look for plateaus
 */
	       if(j > 0 && v00 == rm1[j - 1]) { // a plateau */
		  if(idm1[j - 1]) {	// continued from previous line */
		     id0[j] = idm1[j - 1];
		  } else {
		     if(++next_id == plateau_size) { // out of space */
			next_id--; npeak_max = 0;
			break;
		     }
		     plateau[next_id].alias = 0;
		     plateau[next_id].not_max = 0;			
		     plateau[next_id].n = 0;
		     plateau[next_id].peak = v00;
		     plateau[next_id].rpeak = y - 1;
		     plateau[next_id].cpeak = j - 1;
		     plateau[next_id].row_sum = 0;
		     plateau[next_id].col_sum = 0;
		     id0[j] = next_id;

		     plateau[id0[j]].n++;
		     plateau[id0[j]].row_sum += y - 1;
		     plateau[id0[j]].col_sum += j - 1;
		     plateau[id0[j]].not_max |=
					      not_max(rm2, rm1, r0, j-1, ncol);

		     idm1[j - 1] = id0[j];
		  }
	       }

	       if(v00 == rm1[j]) {	// a plateau */
		  if(idm1[j]) {		// rm1[j] is already on a plateau */
		     if(!id0[j]) {	// r0[j] isn't on a plateau */
			id0[j] = idm1[j];
		     } else if(id0[j] != idm1[j] &&
			       id0[j] != (a = resolve_alias(plateau,idm1[j]))){
			id0[j] = plateau[id0[j]].alias = a;
		     } else {
			;		// already the same plateau */
		     }
		  } else {
		     if(!id0[j]) {	// need a new plateau */
			if(++next_id == plateau_size) {
			   next_id--; npeak_max = 0;
			   break;
			}
			plateau[next_id].alias = 0;
			plateau[next_id].not_max = 0;			
			plateau[next_id].n = 0;
			plateau[next_id].peak = v00;
			plateau[next_id].rpeak = y - 1;
			plateau[next_id].cpeak = j;
			plateau[next_id].row_sum = 0;
			plateau[next_id].col_sum = 0;
			id0[j] = next_id;
		     }
		     plateau[id0[j]].n++;
		     plateau[id0[j]].row_sum += y - 1;
		     plateau[id0[j]].col_sum += j;
		     plateau[id0[j]].not_max |= not_max(rm2, rm1, r0, j, ncol);

		     idm1[j] = id0[j];
		  }
	       }

	       if(j < ncol - 1 && v00 == rm1[j + 1]) { // a plateau */
		  if(idm1[j + 1]) {	// rm1[j+1] is already on a plateau */
		     if(!id0[j]) {	// r0[j] isn't on a plateau */
			id0[j] = idm1[j + 1];
		     } else if(idm1[j + 1] != id0[j] &&
			       idm1[j + 1] !=
					 (a = resolve_alias(plateau,id0[j]))) {
			idm1[j + 1] = plateau[idm1[j + 1]].alias = a;
		     } else {
			;		// already on same plateau */
		     }
		  } else {
		     if(!id0[j]) {	// need a new plateau */
			if(++next_id == plateau_size) {
			   next_id--; npeak_max = 0;
			   break;
			}
			plateau[next_id].alias = 0;
			plateau[next_id].not_max = 0;			
			plateau[next_id].n = 0;
			plateau[next_id].peak = v00;
			plateau[next_id].rpeak = y - 1;
			plateau[next_id].cpeak = j + 1;
			plateau[next_id].row_sum = 0;
			plateau[next_id].col_sum = 0;
			id0[j] = next_id;
		     }
		     plateau[id0[j]].n++;
		     plateau[id0[j]].row_sum += y - 1;
		     plateau[id0[j]].col_sum += j + 1;
		     plateau[id0[j]].not_max |=
					      not_max(rm2, rm1, r0, j+1, ncol);

		     idm1[j + 1] = id0[j];
		  }
	       }
/*
 * done with the previous row; now look at the current one
 */
	       if(j > 0 && v00 == r0[j - 1]) {	// a plateau */
		  if(id0[j - 1]) {	// r0[j-1] is already on a plateau */
		     if(!id0[j]) {	// r0[j] isn't on a plateau */
			id0[j] = id0[j - 1];
		     } else if(id0[j] != id0[j - 1] &&
			       id0[j] !=
				     (a = resolve_alias(plateau,id0[j - 1]))) {
			id0[j] = plateau[id0[j]].alias = a;
		     } else {
			;		// already on same plateau */
		     }
		  } else {
		     if(!id0[j]) {	// need a new plateau */
			if(++next_id == plateau_size) {
			   next_id--; npeak_max = 0;
			   break;
			}
			plateau[next_id].alias = 0;
			plateau[next_id].not_max = 0;			
			plateau[next_id].n = 0;
			plateau[next_id].peak = v00;
			plateau[next_id].rpeak = y;
			plateau[next_id].cpeak = j - 1;
			plateau[next_id].row_sum = 0;
			plateau[next_id].col_sum = 0;
			id0[j] = next_id;
		     }
		     plateau[id0[j]].n++;
		     plateau[id0[j]].row_sum += y;
		     plateau[id0[j]].col_sum += j - 1;
		     plateau[id0[j]].not_max |=
					      not_max(rm1, r0, rp1, j-1, ncol);
		     
		     id0[j - 1] = id0[j];
		  }
	       }

	       if(id0[j]) {
		  plateau[id0[j]].n++;
		  plateau[id0[j]].row_sum += y;
		  plateau[id0[j]].col_sum += j;
		  if(!plateau[id0[j]].not_max) {
		     plateau[id0[j]].not_max |= not_max(rm1, r0, rp1, j, ncol);
		  }
	       }
	    }
	 } while(++i < nspan && spans[i].y - row0 == y); // the same row */
      }
   }
/*
 * process plateau data. Start by resolving aliases
 */
   for(i = 1; i <= next_id;i++) {
      if(plateau[i].n == 0) {		// no pixels in this plateau
					   (probably due to aliasing) */
	 continue;
      }
      
      if((a = resolve_alias(plateau,i)) == i) {
	 continue;			// no aliasing */
      }
      
      shAssert(plateau[a].peak == plateau[i].peak);
      
      plateau[a].n += plateau[i].n;
      plateau[a].row_sum += plateau[i].row_sum;
      plateau[a].col_sum += plateau[i].col_sum;
      plateau[a].not_max |= plateau[i].not_max;

      plateau[i].n = 0;
   }
   
   for(i = 1; i <= next_id;i++) {
      if(plateau[i].n == 0) {		// no pixels in this plateau
					   (probably due to aliasing) */
	 continue;
      }
      if(plateau[i].not_max) {
	 continue;			// not a maximum */
      }

      shAssert(plateau[i].alias == 0);
      
      if(peaks->npeak >= peaks->size) {
	 phPeaksRealloc(peaks, 2*peaks->size);
      }
      peaks->peaks[peaks->npeak]->peak = plateau[i].peak;
      peaks->peaks[peaks->npeak]->rpeak = plateau[i].rpeak;
      peaks->peaks[peaks->npeak]->cpeak = plateau[i].cpeak;
      peaks->peaks[peaks->npeak]->rowc =
			       (float)plateau[i].row_sum/plateau[i].n + 0.4999;
      peaks->peaks[peaks->npeak]->colc =
			       (float)plateau[i].col_sum/plateau[i].n + 0.4999;
      peaks->npeak++;
   }
/*
 * sort the peak list, so the first is the brightest
 */
   phPeaksSort(peaks);
/*
 * see if we picked up the global maximum as one of our peaks. We can in
 * fact pick up a _different_ peak with the same peak intensity as the
 * global maximum that we found, but that's OK
 */
   if(peaks->npeak > 0) {
      if(peaks->peaks[0]->peak == max) { // found a/the max peak */
	 return(peaks->npeak);
      }
   }
/*
 * Oh dear; we have a maximum pixel that isn't on the peak list. It may be
 * a plateau, and we really should deal with that case for consistency.
 * It's not as bad as the full plateau stuff above, as we know that this
 * (possible) plateau's a maximum, but it's still a full object finding
 * problem. At least for now we won't bother with this case.
 *
 * We have to shuffle all the peaks down to make room for this one.
 */
   if(peaks->npeak >= peaks->size) {
      phPeaksRealloc(peaks,peaks->size + 1);
   }
   tmpeak = peaks->peaks[peaks->npeak];
   for(k = peaks->npeak;k > 0;k--) {
      peaks->peaks[k] = peaks->peaks[k - 1];
   }
   peaks->peaks[0] = tmpeak;
   
   peaks->peaks[0]->peak = max;
   peaks->peaks[0]->rpeak = max_y;
   peaks->peaks[0]->cpeak = max_x;
   peaks->peaks[0]->rowc = max_y + 0.4999;
   peaks->peaks[0]->colc = max_x + 0.4999;
   peaks->npeak++;
   
#endif
   return(peaks);
}

}}}
