// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MasterElement_h
#define MasterElement_h

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include "stk_util/util/ReportHandler.hpp"

#include <stdexcept>

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

namespace MEconstants {
static const double realmin = std::numeric_limits<double>::min();
}

namespace Jacobian {
enum Direction { S_DIRECTION = 0, T_DIRECTION = 1, U_DIRECTION = 2 };
}

struct ElementDescription;
class MasterElement;

class MasterElement
{
public:
  KOKKOS_FUNCTION
  MasterElement(const double scaleToStandardIsoFac = 1.0);
  KOKKOS_FUNCTION virtual ~MasterElement() {}

  template <typename SCALAR, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void
  shape_fcn(SharedMemView<SCALAR**, SHMEM>& /* shpfc */);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void
  shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& /* shpfc */);

  KOKKOS_FUNCTION virtual void grad_op(
    const SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::grad_op not implemented for element");
  }

  virtual void grad_op(
    const SharedMemView<double**>& /* coords */,
    SharedMemView<double***>& /* gradop */,
    SharedMemView<double***>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::grad_op not implemented for element");
  }

  KOKKOS_FUNCTION virtual void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::shifted_grad_op not implemented for element");
  }

  KOKKOS_FUNCTION virtual void face_grad_op(
    int /* face_ordinal */,
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::face_grad_op not implemented for element");
  }

  KOKKOS_FUNCTION virtual void shifted_face_grad_op(
    int /* face_ordinal */,
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::shifted_face_grad_op not implemented for element");
  }

  KOKKOS_FUNCTION virtual void grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */,
    SharedMemView<DoubleType*, DeviceShmem>& /*det_j*/)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::grad_op_fem not implemented for element");
  }

  KOKKOS_FUNCTION virtual void shifted_grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */,
    SharedMemView<DoubleType*, DeviceShmem>& /*det_j*/)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::shifted_grad_op_fem not implemented for element");
  }

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType**, DeviceShmem>& /* areav */)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::determinant not implemented for element: "
                      "DoubleType area");
  }

  virtual void determinant(
    const SharedMemView<double**>& /* coords */,
    SharedMemView<double**>& /* areav */)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::determinant not implemented for element: double area");
  }

  KOKKOS_FUNCTION virtual void gij(
    const SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gupper */,
    SharedMemView<DoubleType***, DeviceShmem>& /* glower */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::gij not implemented for element");
  }

  KOKKOS_FUNCTION virtual void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* metric */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::Mij not implemented for element");
  }

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType*, DeviceShmem>& /* volume */)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::determinant not implemented for element: "
                      "DoubleType volume");
  }

  virtual void determinant(
    const SharedMemView<double**>& /* coords */,
    SharedMemView<double*>& /* volume */)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::determinant not implemented for element: double volume");
  }

  virtual void
  Mij(const double* /* coords */, double* /* metric */, double* /* deriv */)
  {
    throw std::runtime_error("Mij not implemented");
  }

  virtual void
  nodal_grad_op(const int /* nelem */, double* /* deriv */, double* /* error */)
  {
    throw std::runtime_error("nodal_grad_op not implemented");
  }

  KOKKOS_FUNCTION virtual const int* adjacentNodes()
  {
    STK_NGP_ThrowErrorMsg("MasterElement::adjacentNodes not implemented");
    return nullptr;
  }

  KOKKOS_FUNCTION virtual const int* scsIpEdgeOrd()
  {
    STK_NGP_ThrowErrorMsg("MasterElement::scsIpEdgeOrd not implemented");
    return nullptr;
  }

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int /* ordinal */ = 0) const
  {
    STK_NGP_ThrowErrorMsg("MasterElement::ipNodeMap not implemented");
    return nullptr;
  }

  KOKKOS_FUNCTION virtual int
  opposingNodes(const int /* ordinal */, const int /* node */)
  {
    STK_NGP_ThrowErrorMsg("opposingNodes not implemented");
    return -1;
  }

  KOKKOS_FUNCTION virtual int
  opposingFace(const int /* ordinal */, const int /* node */)
  {
    STK_NGP_ThrowErrorMsg("opposingFace not implemented");
    return -1;
  }

  virtual double isInElement(
    const double* /* elemNodalCoord */,
    const double* /* pointCoord */,
    double* /* isoParCoord */)
  {
    throw std::runtime_error("isInElement not implemented");
  }

  virtual void interpolatePoint(
    const int& /* nComp */,
    const double* /* isoParCoord */,
    const double* /* field */,
    double* /* result */)
  {
    throw std::runtime_error("interpolatePoint not implemented");
  }

  virtual void general_shape_fcn(
    const int /* numIp */, const double* /* isoParCoord */, double* /* shpfc */)
  {
    throw std::runtime_error("general_shape_fcn not implement");
  }

  virtual void general_face_grad_op(
    const int /* face_ordinal */,
    const double* /* isoParCoord */,
    const double* /* coords */,
    double* /* gradop */,
    double* /* det_j */,
    double* /* error  */)
  {
    throw std::runtime_error("general_face_grad_op not implemented");
  }

  virtual void general_normal(
    const double* /* isoParCoord */,
    const double* /* coords */,
    double* /* normal */)
  {
    throw std::runtime_error("general_normal not implemented");
  }

  virtual void sidePcoords_to_elemPcoords(
    const int& /* side_ordinal */,
    const int& /* npoints */,
    const double* /* side_pcoords */,
    double* /* elem_pcoords */)
  {
    throw std::runtime_error("sidePcoords_to_elemPcoords");
  }

  double
  isoparametric_mapping(const double b, const double a, const double xi) const;
  bool within_tolerance(const double& val, const double& tol) const;
  double vector_norm_sq(const double* vect, int len) const;

  virtual int ndim() const { return nDim_; }
  virtual int nodes_per_element() const { return nodesPerElement_; }
  KOKKOS_FUNCTION int num_integration_points() const { return numIntPoints_; }
  double scal_to_standard_iso_factor() const { return scaleToStandardIsoFac_; }

  KOKKOS_FUNCTION virtual const int* adjacentNodes() const
  {
#if !defined(KOKKOS_ENABLE_GPU)
    throw std::runtime_error("adjacentNodes not implimented");
#else
    return nullptr;
#endif
  }
  virtual const double* integration_locations() const
  {
    throw std::runtime_error("integration_locations not implemented");
  }
  virtual const double* integration_location_shift() const
  {
    throw std::runtime_error("adjacentNodes not implimented");
  }
  virtual const double* integration_exp_face_shift() const
  {
    throw std::runtime_error("integration_exp_face_shift not implimented");
  }
  KOKKOS_FUNCTION virtual const int* side_node_ordinals(int) const
  {
    STK_NGP_ThrowErrorMsg("side_node_ordinals not implemented");
    return nullptr;
  }

  int nDim_;
  int nodesPerElement_;

protected:
  int numIntPoints_;

  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>&)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::shape_fcn not implemented for element");
  }
  virtual void shape_fcn(SharedMemView<double**, HostShmem>&)
  {
    STK_NGP_ThrowErrorMsg("MasterElement::shape_fcn not implemented for element");
  }

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>&)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::shifted_shape_fcn not implemented for element");
  }
  virtual void shifted_shape_fcn(SharedMemView<double**, HostShmem>&)
  {
    STK_NGP_ThrowErrorMsg(
      "MasterElement::shifted_shape_fcn not implemented for element");
  }

private:
  const double scaleToStandardIsoFac_;
};

template <>
KOKKOS_INLINE_FUNCTION void
MasterElement::shape_fcn<DoubleType, DeviceShmem>(
  SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn(shpfc);
}

template <>
KOKKOS_INLINE_FUNCTION void
MasterElement::shape_fcn<double, HostShmem>(
  SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn(shpfc);
}

template <>
KOKKOS_INLINE_FUNCTION void
MasterElement::shifted_shape_fcn<DoubleType, DeviceShmem>(
  SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn(shpfc);
}

template <>
KOKKOS_INLINE_FUNCTION void
MasterElement::shifted_shape_fcn<double, HostShmem>(
  SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn(shpfc);
}

} // namespace nalu
} // namespace sierra

#endif
