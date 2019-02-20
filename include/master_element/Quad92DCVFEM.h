/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Quad92DCVFEM_h  
#define Quad92DCVFEM_h  

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>

namespace sierra{
namespace nalu{

// 3D Quad 27 subcontrol volume
class Quad92DSCV : public QuadrilateralP2Element
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;

  Quad92DSCV();
  virtual ~Quad92DSCV() {}

  const int * ipNodeMap(int ordinal = 0) override ;

  void determinant(
    SharedMemView<DoubleType**> &coords,
    SharedMemView<DoubleType*> &vol) override ;

  void grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void shifted_grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void Mij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override ;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override ;

  void Mij(
    const double *coords,
    double *metric,
    double *deriv) override ;

private:
  void set_interior_info();

  DoubleType jacobian_determinant(
    const SharedMemView<DoubleType**> &coords,
    const double *POINTER_RESTRICT shapeDerivs ) const;

  double jacobian_determinant(
    const double *POINTER_RESTRICT elemNodalCoords,
    const double *POINTER_RESTRICT shapeDerivs ) const;

  std::vector<double> ipWeight_;
};

// 3D Hex 27 subcontrol surface
class Quad92DSCS : public QuadrilateralP2Element
{
public:
  using MasterElement::determinant;

  Quad92DSCS();
  virtual ~Quad92DSCS() {}

  void determinant(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType**>& areav) override ;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override ;

  void grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error ) override ;

  void shifted_grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void shifted_grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error ) override ;


  void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error ) override ;

  void gij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv) override ;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) override ;

  void Mij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override ;

  void Mij(
    const double *coords,
    double *metric,
    double *deriv) override ;

  const int * adjacentNodes() override ;

  const int * ipNodeMap(int ordinal = 0) override ;

  int opposingNodes(
    const int ordinal, const int node) override ;

  int opposingFace(
    const int ordinal, const int node) override ;

  const int* side_node_ordinals(int sideOrdinal) final;
  virtual const std::vector<int>& side_node_ordinals() const final {return sideNodeOrdinals_;};
  virtual void side_node_ordinals(const std::vector<int>& v) final {sideNodeOrdinals_=v;};


private:
  void set_interior_info();
  void set_boundary_info();

  template <Jacobian::Direction direction> void
  area_vector(
    const SharedMemView<DoubleType**>& elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    DoubleType *POINTER_RESTRICT areaVector ) const;
  template <Jacobian::Direction direction> void
  area_vector(
    const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT areaVector ) const;

  std::vector<ContourData> ipInfo_;
  int ipsPerFace_;
};

} // namespace nalu
} // namespace Sierra

#endif
