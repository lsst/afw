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
 * \file
 * \brief An include file to include the public header files for lsst::afw::math
 *
 * Note: this header file intentionally excludes math/detail header files because they define
 * classes and functions which are not part of the public API.
 */
#ifndef LSST_AFW_MATH_H
#define LSST_AFW_MATH_H

#include "lsst/afw/math/GaussianProcess.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
#include "lsst/afw/math/minimize.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/math/SpatialCell.h"
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/math/MaskedVector.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Integrate.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Random.h"
#include "lsst/afw/math/LeastSquares.h"

#endif // LSST_AFW_MATH_H
