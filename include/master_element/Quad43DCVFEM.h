/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Quad43DSCS_h
#define Quad43DSCS_h

#include "master_element/MasterElement.h"

#include <array>

namespace sierra{
namespace nalu{

// 3D Quad 4
class Quad3DSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4;
  using MasterElement::determinant;

  KOKKOS_FUNCTION
  Quad3DSCS();
  KOKKOS_FUNCTION
  virtual ~Quad3DSCS() = default;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;
 
  // NGP-ready methods first
  KOKKOS_FUNCTION void shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &shpfc) override;

  KOKKOS_FUNCTION void shifted_shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &shpfc) override;

  KOKKOS_FUNCTION void determinant(
    SharedMemView<DoubleType**, DeviceShmem>&coords,
    SharedMemView<DoubleType**, DeviceShmem>&areav) override;

  KOKKOS_FUNCTION void quad4_shape_fcn(
    const double *isoParCoord,
    SharedMemView<DoubleType**, DeviceShmem> &shpfc);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override;

  void shape_fcn(
    double *shpfc) override;

  void shifted_shape_fcn(
    double *shpfc) override;

  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord) override;
  
  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result) override;

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc) override;

  void general_normal(
    const double *isoParCoord,
    const double *coords,
    double *normal) override;

  void non_unit_face_normal(
    const double * par_coord,
    const double * elem_nodal_coor,
    double * normal_vector );
  
  double parametric_distance(const std::array<double,3> &x);

  const double elemThickness_;


  virtual const double* integration_locations() const final {
    return intgLoc_;
  }
  virtual const double* integration_location_shift() const final {
    return intgLocShift_;
  }

private:

  static const int nDim_ = AlgTraits::nDim_;
  static const int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static const int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr double scaleToStandardIsoFac_ = 2.0;

  // define ip node mappings; ordinal size = 1
  const int ipNodeMap_[nodesPerElement_] = {0, 1, 2, 3};

  // standard integration location
  const double intgLoc_[8] = { 
   -0.25,  -0.25, // surf 1
    0.25,  -0.25, // surf 2
    0.25,   0.25, // surf 3
   -0.25,   0.25};// surf 4

  // shifted
  const double intgLocShift_[8] = { 
   -0.50,  -0.50, // surf 1
    0.50,  -0.50, // surf 2
    0.50,   0.50, // surf 3
   -0.50,   0.50};// surf 4  


};
    
} // namespace nalu
} // namespace Sierra

#endif
