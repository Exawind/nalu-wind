// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Tet4CVFEM_h
#define Tet4CVFEM_h

#include <master_element/MasterElement.h>

namespace sierra {
namespace nalu {

// Tet 4 subcontrol volume
class TetSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsTet4;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  TetSCV();
  KOKKOS_FUNCTION virtual ~TetSCV() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

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

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void tet_shape_fcn(
    const int npts,
    const double* par_coord,
    SharedMemView<SCALAR**, SHMEM>& shpfc) const;

  virtual const double* integration_locations() const final
  {
    return &intgLoc_[0][00];
  }
  virtual const double* integration_location_shift() const final
  {
    return &intgLocShift_[0][0];
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
  const double seventeen96ths = 17.0 / 96.0;
  const double fourfive96ths = 45.0 / 96.0;
  const double intgLoc_[4][3] = {
    {seventeen96ths, seventeen96ths, seventeen96ths}, // vol 1
    {fourfive96ths, seventeen96ths, seventeen96ths},  // vol 2
    {seventeen96ths, fourfive96ths, seventeen96ths},  // vol 3
    {seventeen96ths, seventeen96ths, fourfive96ths}}; // vol 4

  // shifted
  const double intgLocShift_[4][3] = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scv(
    const SharedMemView<DBLTYPE**, SHMEM>& coordel,
    SharedMemView<DBLTYPE*, SHMEM>& volume) const;
};

// Tet 4 subcontrol surface
class TetSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsTet4;
  using MasterElement::adjacentNodes;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  TetSCS();
  KOKKOS_FUNCTION virtual ~TetSCS() {}

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

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void tet_shape_fcn(
    const int npts,
    const double* par_coord,
    SharedMemView<SCALAR**, SHMEM>& shpfc) const;

  void
  tet_shape_fcn(const int npts, const double* par_coord, double* shape_fcn);

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

  double parametric_distance(const double* x);

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

  const int sideNodeOrdinals_[4][3] = {
    {0, 1, 3}, // ordinal 0
    {1, 2, 3}, // ordinal 1
    {0, 3, 2}, // ordinal 2
    {0, 2, 1}  // ordinal 3
  };

  // define L/R mappings
  const int lrscv_[12] = {0, 1, 1, 2, 0, 2, 0, 3, 1, 3, 2, 3};

  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[6] = {0, 1, 2, 3, 4, 5};

  // define opposing node
  const int oppNode_[4][3]{
    {2, 2, 2},  // face 0
    {0, 0, 0},  // face 1
    {1, 1, 1},  // face 2
    {3, 3, 3}}; // face 3

  // define opposing face
  const int oppFace_[4][3] = {
    {2, 1, 5},  // face 0
    {0, 2, 3},  // face 1
    {0, 4, 1},  // face 2
    {3, 5, 4}}; // face 3

  // standard integration location
  const double thirteen36ths = 13.0 / 36.0;
  const double five36ths = 5.0 / 36.0;
  const double intgLoc_[18] = {
    thirteen36ths, five36ths,     five36ths,      // surf 1    1->2
    thirteen36ths, thirteen36ths, five36ths,      // surf 2    2->3
    five36ths,     thirteen36ths, five36ths,      // surf 3    1->3
    five36ths,     five36ths,     thirteen36ths,  // surf 4    1->4
    thirteen36ths, five36ths,     thirteen36ths,  // surf 5    2->4
    five36ths,     thirteen36ths, thirteen36ths}; // surf 6    3->4

  // shifted
  const double intgLocShift_[18] = {0.50, 0.00, 0.00,  // surf 1    1->2
                                    0.50, 0.50, 0.00,  // surf 2    2->3
                                    0.00, 0.50, 0.00,  // surf 3    1->3
                                    0.00, 0.00, 0.50,  // surf 4    1->4
                                    0.50, 0.00, 0.50,  // surf 5    2->4
                                    0.00, 0.50, 0.50}; // surf 6    3->4

#if 0
  // exposed face
  const double seven36ths = 7.0/36.0;
  const double eleven18ths = 11.0/18.0;
  const double intgExpFace_[4][3][3] = {
  // face 0: nodes 0,1,3: scs 0, 1, 2
  {{seven36ths,   0.00,  seven36ths},
   {eleven18ths,  0.00,  seven36ths},
   {seven36ths,   0.00,  eleven18ths}},
  // face 1: nodes 1,2,3: scs 0, 1, 2
  {{eleven18ths,  seven36ths,   seven36ths},
   {seven36ths,   eleven18ths,  seven36ths},
   {seven36ths,   seven36ths,   eleven18ths}},
  // face 2: nodes 0,3,2: scs 0, 1, 2
  {{0.00,       seven36ths,   seven36ths},
   {0.00,       seven36ths,   eleven18ths},
   {0.00,       eleven18ths,  seven36ths}},
  //face 3: nodes 0, 2, 1: scs 0, 1, 2
  {{seven36ths,   seven36ths,    0.00},
   {eleven18ths,  seven36ths,    0.00},
   {seven36ths,   eleven18ths,   0.00}}};
#endif

  // boundary integration point ip node mapping (ip on an ordinal to local node
  // number)
  const int ipNodeMap_[4][3] = {
    // 3 ips * 4 faces
    {0, 1, 3},  // face 0
    {1, 2, 3},  // face 1
    {0, 3, 2},  // face 2
    {0, 2, 1}}; // face 3

#if !defined(KOKKOS_ENABLE_GPU)
  int intgExpFaceShift_[4][3][3];
#endif

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scs(
    const SharedMemView<DBLTYPE**, SHMEM>& coordel,
    SharedMemView<DBLTYPE**, SHMEM>& areav) const;
};

} // namespace nalu
} // namespace sierra

#endif
