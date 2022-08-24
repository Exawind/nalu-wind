// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FORTRAN_Proto_h
#define FORTRAN_Proto_h

#include <stk_util/util/Fortran.hpp>

//------------------------------------------------------------------------------
//---------------------- WORKSET ALGORITHMS ------------------------------------
//------------------------------------------------------------------------------

extern "C" void SIERRA_FORTRAN(hex_shape_fcn)(
  const int* npts, const double* par_coords, double* shape_fcn);

extern "C" void SIERRA_FORTRAN(hex_derivative)(
  const int* npts, const double* par_coords, double* deriv);
extern "C" void SIERRA_FORTRAN(quad_derivative)(
  const int* npts, const double* par_coords, double* deriv);

extern "C" void SIERRA_FORTRAN(quad92d_derivative)(
  const int* npts, const double* par_coords, double* deriv);

extern "C" void SIERRA_FORTRAN(tet_derivative)(const int* npts, double* deriv);

extern "C" void SIERRA_FORTRAN(tri_derivative)(const int* npts, double* deriv);

extern "C" void SIERRA_FORTRAN(hex_gradient_operator)(
  const int* nelem,
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error,
  int* lerr);

extern "C" void SIERRA_FORTRAN(quad_gradient_operator)(
  const int* nelem,
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error,
  int* lerr);

extern "C" void SIERRA_FORTRAN(twod_gij)(
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gupperij,
  double* glowerij);

extern "C" void SIERRA_FORTRAN(threed_gij)(
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gupperij,
  double* glowerij);

extern "C" void SIERRA_FORTRAN(tet_gradient_operator)(
  const int* nelem,
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error,
  int* lerr);

extern "C" void SIERRA_FORTRAN(tri_gradient_operator)(
  const int* nelem,
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error,
  int* lerr);

extern "C" void SIERRA_FORTRAN(pyr_gradient_operator)(
  const int* nelem,
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error,
  int* lerr);

extern "C" void SIERRA_FORTRAN(wed_gradient_operator)(
  const int* nelem,
  const int* npe,
  const int* numint,
  double* deriv,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error,
  int* lerr);

extern "C" void SIERRA_FORTRAN(quad3d_shape_fcn)(
  const int* npts, const double* par_coords, double* shape_fcn);

#endif // FORTRAN_Proto_h
