// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Hex8CVFEM_h
#define Hex8CVFEM_h

#include <array>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>

namespace sierra {
namespace nalu {

// Hex 8 subcontrol volume
class HexSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;

  KOKKOS_FUNCTION
  HexSCV();

  KOKKOS_FUNCTION virtual ~HexSCV() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType*, DeviceShmem>& volume) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double*>& volume) override;

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

  KOKKOS_FUNCTION void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& metric,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void Mij(const double* coords, double* metric, double* deriv) override;

  template <typename ViewType>
  KOKKOS_FUNCTION void shape_fcn(ViewType& shpfc);

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScvIp_;

  // define ip node mappings
  const int ipNodeMap_[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // standard integration location
  const double intgLoc_[numIntPoints_ * nDim_] = {
    -0.25, -0.25, -0.25, +0.25, -0.25, -0.25, +0.25, +0.25,
    -0.25, -0.25, +0.25, -0.25, -0.25, -0.25, +0.25, +0.25,
    -0.25, +0.25, +0.25, +0.25, +0.25, -0.25, +0.25, +0.25};

  // shifted integration location
  const double intgLocShift_[24] = {
    -0.5, -0.5, -0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5, -0.5, +0.5, -0.5,
    -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, +0.5, +0.5, +0.5, -0.5, +0.5, +0.5};

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scv(
    const SharedMemView<DBLTYPE**, SHMEM>& coords,
    SharedMemView<DBLTYPE*, SHMEM>& volume) const;
};

// Hex 8 subcontrol surface
class HexSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;
  using AlgTraitsFace = AlgTraitsQuad4;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  HexSCS();

  KOKKOS_FUNCTION virtual ~HexSCS() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  using MasterElement::determinant;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  KOKKOS_FUNCTION void hex8_gradient_operator(
    const int nodesPerElem,
    const int numIntgPts,
    SharedMemView<DoubleType***, DeviceShmem>& deriv,
    SharedMemView<DoubleType**, DeviceShmem>& cordel,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType*, DeviceShmem>& det_j,
    DoubleType& error,
    int& lerr);

  template <typename ViewTypeCoord, typename ViewTypeGrad>
  KOKKOS_FUNCTION void
  grad_op(ViewTypeCoord& coords, ViewTypeGrad& gradop, ViewTypeGrad& deriv);

  KOKKOS_FUNCTION void grad_op(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void grad_op(
    const SharedMemView<double**>& coords,
    SharedMemView<double***>& gradop,
    SharedMemView<double***>& deriv) override;

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

  KOKKOS_FUNCTION void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType**, DeviceShmem>& areav) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double**>& areav) override;

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
  using MasterElement::side_node_ordinals;

  double parametric_distance(const std::array<double, 3>& x);

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr double scaleToStandardIsoFac_ = 2.0;

  // standard integration location
  const double intgLoc_[numIntPoints_ * nDim_] = {
    0.00,  -0.25, -0.25, // surf 1    1->2
    0.25,  0.00,  -0.25, // surf 2    2->3
    0.00,  0.25,  -0.25, // surf 3    3->4
    -0.25, 0.00,  -0.25, // surf 4    1->4
    0.00,  -0.25, 0.25,  // surf 5    5->6
    0.25,  0.00,  0.25,  // surf 6    6->7
    0.00,  0.25,  0.25,  // surf 7    7->8
    -0.25, 0.00,  0.25,  // surf 8    5->8
    -0.25, -0.25, 0.00,  // surf 9    1->5
    0.25,  -0.25, 0.00,  // surf 10   2->6
    0.25,  0.25,  0.00,  // surf 11   3->7
    -0.25, 0.25,  0.00}; // surf 12   4->8

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  // define L/R mappings
  const int lrscv_[24] = {0, 1, 1, 2, 2, 3, 0, 3, 4, 5, 5, 6,
                          6, 7, 4, 7, 0, 4, 1, 5, 2, 6, 3, 7};

  // boundary integration point ip node mapping (ip on an ordinal to local node
  // number)
  const int ipNodeMap_[24] = {/* face 0 */ 0, 1, 5, 4,
                              /* face 1 */ 1, 2, 6, 5,
                              /* face 2 */ 2, 3, 7, 6,
                              /* face 3 */ 0, 4, 7, 3,
                              /* face 4 */ 0, 3, 2, 1,
                              /* face 5 */ 4, 5, 6, 7};

  // define opposing node
  const int oppNode_[24] = {/* face 0 */ 3, 2, 6, 7,
                            /* face 1 */ 0, 3, 7, 4,
                            /* face 2 */ 1, 0, 4, 5,
                            /* face 3 */ 1, 5, 6, 2,
                            /* face 4 */ 4, 7, 6, 5,
                            /* face 5 */ 0, 1, 2, 3};

  // define opposing face
  const int oppFace_[24] = {/* face 0 */ 3, 1,  5,  7,
                            /* face 1 */ 0, 2,  6,  4,
                            /* face 2 */ 1, 3,  7,  5,
                            /* face 3 */ 0, 4,  6,  2,
                            /* face 4 */ 8, 11, 10, 9,
                            /* face 5 */ 8, 9,  10, 11};

  // shifted
  const double intgLocShift_[36] = {0.00,  -0.50, -0.50, // surf 1    1->2
                                    0.50,  0.00,  -0.50, // surf 2    2->3
                                    0.00,  0.50,  -0.50, // surf 3    3->4
                                    -0.50, 0.00,  -0.50, // surf 4    1->4
                                    0.00,  -0.50, 0.50,  // surf 5    5->6
                                    0.50,  0.00,  0.50,  // surf 6    6->7
                                    0.00,  0.50,  0.50,  // surf 7    7->8
                                    -0.50, 0.00,  0.50,  // surf 8    5->8
                                    -0.50, -0.50, 0.00,  // surf 9    1->5
                                    0.50,  -0.50, 0.00,  // surf 10   2->6
                                    0.50,  0.50,  0.00,  // surf 11   3->7
                                    -0.50, 0.50,  0.00}; // surf 12   4->8

  // exposed face
  const double intgExpFace_[6][4][3] = {
    // face 0; scs 0, 1, 2, 3
    {{-0.25, -0.50, -0.25},
     {0.25, -0.50, -0.25},
     {0.25, -0.50, 0.25},
     {-0.25, -0.50, 0.25}},
    // face 1; scs 0, 1, 2, 3
    {{0.50, -0.25, -0.25},
     {0.50, 0.25, -0.25},
     {0.50, 0.25, 0.25},
     {0.50, -0.25, 0.25}},
    // face 2; scs 0, 1, 2, 3
    {{0.25, 0.50, -0.25},
     {-0.25, 0.50, -0.25},
     {-0.25, 0.50, 0.25},
     {0.25, 0.50, 0.25}},
    // face 3; scs 0, 1, 2, 3
    {{-0.50, -0.25, -0.25},
     {-0.50, -0.25, 0.25},
     {-0.50, 0.25, 0.25},
     {-0.50, 0.25, -0.25}},
    // face 4; scs 0, 1, 2, 3
    {{-0.25, -0.25, -0.50},
     {-0.25, 0.25, -0.50},
     {0.25, 0.25, -0.50},
     {0.25, -0.25, -0.50}},
    // face 5; scs 0, 1, 2, 3
    {{-0.25, -0.25, 0.50},
     {0.25, -0.25, 0.50},
     {0.25, 0.25, 0.50},
     {-0.25, 0.25, 0.50}}};

  const double intgExpFaceShift_[6][4][3] = {
    {{-0.5, -0.5, -0.5},
     {0.5, -0.5, -0.5},
     {0.5, -0.5, 0.5},
     {-0.5, -0.5, 0.5}},
    {{0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}},
    {{0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}},
    {{-0.5, -0.5, -0.5},
     {-0.5, -0.5, 0.5},
     {-0.5, 0.5, 0.5},
     {-0.5, 0.5, -0.5}},
    {{-0.5, -0.5, -0.5},
     {-0.5, 0.5, -0.5},
     {0.5, 0.5, -0.5},
     {0.5, -0.5, -0.5}},
    {{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5}}};

  // nodes for collocation calculations
  const double nodeLoc_[8][3] = {/* node 0 */ {-0.5, -0.5, -0.5},
                                 /* node 1 */ {0.5, -0.5, -0.5},
                                 /* node 2 */ {0.5, 0.5, -0.5},
                                 /* node 3 */ {-0.5, 0.5, -0.5},
                                 /* node 4 */ {-0.5, -0.5, 0.5},
                                 /* node 5 */ {0.5, -0.5, 0.5},
                                 /* node 6 */ {0.5, 0.5, 0.5},
                                 /* node 7 */ {-0.5, 0.5, 0.5}};

  // mapping from a side ordinal to the node ordinals on that side
  const int sideNodeOrdinals_[6][4] = {
    {0, 1, 5, 4},  // ordinal 0
    {1, 2, 6, 5},  // ordinal 1
    {2, 3, 7, 6},  // ordinal 2
    {0, 4, 7, 3},  // ordinal 3
    {0, 3, 2, 1},  // ordinal 4
    {4, 5, 6, 7}}; // ordinal 5

  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

private:
  template <bool shifted>
  KOKKOS_FUNCTION void face_grad_op_t(
    const int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv);

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scs(
    const SharedMemView<DBLTYPE**, SHMEM>& coords,
    SharedMemView<DBLTYPE**, SHMEM>& areav) const;
};

template <typename ViewTypeCoord, typename ViewTypeGrad>
KOKKOS_FUNCTION void
HexSCS::grad_op(
  ViewTypeCoord& coords, ViewTypeGrad& gradop, ViewTypeGrad& deriv)
{
  const SharedMemView<const double**, HostShmem> par_coord(
    intgLoc_, numIntPoints_, nDim_);
  hex8_derivative(intgLoc_, deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

} // namespace nalu
} // namespace sierra

#endif
