/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Quad43DSCS_h
#define Quad43DSCS_h

#include "master_element/MasterElement.h"

namespace sierra{
namespace nalu{

// 3D Quad 4
class Quad3DSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4;
  using MasterElement::determinant;

  Quad3DSCS();
  virtual ~Quad3DSCS();

  const int * ipNodeMap(int ordinal = 0);
 
  // NGP-ready methods first
  void shape_fcn(
    SharedMemView<DoubleType**> &shpfc);

  void shifted_shape_fcn(
    SharedMemView<DoubleType**> &shpfc);

  void quad4_shape_fcn(
    const int  &numIp,
    const double *isoParCoord,
    SharedMemView<DoubleType**> &shpfc);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord);
  
  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc);

  void general_normal(
    const double *isoParCoord,
    const double *coords,
    double *normal);

  void non_unit_face_normal(
    const double * par_coord,
    const double * elem_nodal_coor,
    double * normal_vector );
  
  double parametric_distance(const std::vector<double> &x);

  const double elemThickness_;
};
    
} // namespace nalu
} // namespace Sierra

#endif
