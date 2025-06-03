// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Quad42DCVFEM_h
#define Quad42DCVFEM_h

#include <master_element/MasterElement.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>

namespace sierra {
namespace nalu {

// 2D Quad 4 subcontrol volume
class Quad42DSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4_2D;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  Quad42DSCV();
  KOKKOS_FUNCTION virtual ~Quad42DSCV() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType*, DeviceShmem>& vol) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double*>& vol) override;

  KOKKOS_FUNCTION void grad_op(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& metric,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void Mij(const double* coords, double* metric, double* deriv) override;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScvIp_;

  // define ip node mappings
  const int ipNodeMap_[4] = {0, 1, 2, 3};

  // standard integration location
  const double intgLoc_[8] = {-0.25, -0.25, +0.25, -0.25,
                              +0.25, +0.25, -0.25, +0.25};

  // shifted integration location
  const double intgLocShift_[8] = {-0.50, -0.50, +0.50, -0.50,
                                   +0.50, +0.50, -0.50, +0.50};

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void quad_shape_fcn(
    const double* par_coord, SharedMemView<SCALAR**, SHMEM>& shape_fcn);

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scv(
    const SharedMemView<DBLTYPE**, SHMEM>& coords,
    SharedMemView<DBLTYPE*, SHMEM>& vol);
};

// 2D Quad 4 subcontrol surface
class Quad42DSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4_2D;
  using MasterElement::adjacentNodes;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  Quad42DSCS();
  KOKKOS_FUNCTION virtual ~Quad42DSCS() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType**, DeviceShmem>& areav) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double**>& areav) override;

  KOKKOS_FUNCTION void grad_op(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void grad_op(
    const SharedMemView<double**>& coords,
    SharedMemView<double***>& gradop,
    SharedMemView<double***>& deriv) override;

  KOKKOS_FUNCTION void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) final;

  KOKKOS_FUNCTION void shifted_face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) final;

  KOKKOS_FUNCTION void gij(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gupper,
    SharedMemView<DoubleType***, DeviceShmem>& glower,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& metric,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void Mij(const double* coords, double* metric, double* deriv) override;

  KOKKOS_FUNCTION virtual const int* adjacentNodes() final;

  KOKKOS_FUNCTION virtual const int* scsIpEdgeOrd() final;

  KOKKOS_FUNCTION int opposingNodes(const int ordinal, const int node) override;

  KOKKOS_FUNCTION int opposingFace(const int ordinal, const int node) override;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  double isInElement(
    const double* elemNodalCoord,
    const double* pointCoord,
    double* isoParCoord) override;

  void interpolatePoint(
    const int& nComp,
    const double* isoParCoord,
    const double* field,
    double* result) override;

  void general_shape_fcn(
    const int numIp, const double* isoParCoord, double* shpfc) override;

  void general_face_grad_op(
    const int face_ordinal,
    const double* isoParCoord,
    const double* coords,
    double* gradop,
    double* det_j,
    double* error) override;

  void sidePcoords_to_elemPcoords(
    const int& side_ordinal,
    const int& npoints,
    const double* side_pcoords,
    double* elem_pcoords) override;

  KOKKOS_FUNCTION const int* side_node_ordinals(int sideOrdinal) const final;

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr double scaleToStandardIsoFac_ = 2.0;

  // define L/R mappings
  const int lrscv_[8] = {0, 1, 1, 2, 2, 3, 0, 3};

  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[numIntPoints_] = {0, 1, 2, 3};

  // define opposing node
  const int oppNode_[4][2] = {
    {3, 2},  // face 0; nodes 0,1
    {0, 3},  // face 1; nodes 1,2
    {1, 0},  // face 2; nodes 2,3
    {2, 1}}; // face 3; nodes 3,0

  // define opposing face
  const int oppFace_[4][2] = {
    {3, 1},  // face 0
    {0, 2},  // face 1
    {1, 3},  // face 2
    {2, 0}}; // face 3

  // standard integration location
  const double intgLoc_[8] = {0.00,  -0.25, // surf 1; 1->2
                              0.25,  0.00,  // surf 2; 2->3
                              0.00,  0.25,  // surf 3; 3->4
                              -0.25, 0.00}; // surf 3; 1->5

  // shifted
  const double intgLocShift_[8] = {0.00, -0.50, 0.50,  0.00,
                                   0.00, 0.50,  -0.50, 0.00};

  // exposed face
  const double intgExpFace_[4][2][2] = {
    {{-0.25, -0.50}, {0.25, -0.50}},  // face 0; scs 0, 1; nodes 0,1
    {{0.50, -0.25}, {0.50, 0.25}},    // face 1; scs 0, 1; nodes 1,2
    {{0.25, 0.50}, {-0.25, 0.50}},    // face 2, surf 0, 1; nodes 2,3
    {{-0.50, 0.25}, {-0.50, -0.25}}}; // face 3, surf 0, 1; nodes 3,0

  // boundary integration point ip node mapping (ip on an ordinal to local node
  // number)
  const int ipNodeMap_[4][2] = {
    // 2 ips * 4 faces
    {0, 1},  // face 0;
    {1, 2},  // face 1;
    {2, 3},  // face 2;
    {3, 0}}; // face 3;

  const int sideNodeOrdinals_[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

  double intgExpFaceShift_[4][2][2];

  KOKKOS_FUNCTION
  void face_grad_op(
    const int face_ordinal,
    const bool shifted,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void quad_shape_fcn(
    const double* par_coord, SharedMemView<SCALAR**, SHMEM>& shape);

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scs(
    const SharedMemView<DBLTYPE**, SHMEM>& coords,
    SharedMemView<DBLTYPE**, SHMEM>& areav) const;
};

} // namespace nalu
} // namespace sierra

#endif
