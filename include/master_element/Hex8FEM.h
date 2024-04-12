// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Hex8FEM_h
#define Hex8FEM_h

#include <master_element/MasterElement.h>

namespace sierra {
namespace nalu {

// Hex 8 FEM; -1.0 : 1.0 range
class Hex8FEM : public MasterElement
{
public:
  KOKKOS_FUNCTION
  Hex8FEM();
  KOKKOS_FUNCTION virtual ~Hex8FEM() {}

  using AlgTraits = AlgTraitsHex8;
  using MasterElement::face_grad_op;
  using MasterElement::gij;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION void grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv,
    SharedMemView<DoubleType*, DeviceShmem>& det_j) final;

  KOKKOS_FUNCTION void shifted_grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv,
    SharedMemView<DoubleType*, DeviceShmem>& det_j) final;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  void general_shape_fcn(
    const int numIp, const double* isoParCoord, double* shpfc) override;

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

  // weights; -1:1
  double weights_[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  static const int nDim_ = AlgTraits::nDim_;
  static const int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static const int numIntPoints_ = AlgTraits::numScvIp_;

  // shifted to nodes (Gauss Lobatto)
  const double glIP = 1.0;
  const double intgLocShift_[24] = {-glIP, -glIP, -glIP, +glIP, -glIP, -glIP,
                                    +glIP, +glIP, -glIP, -glIP, +glIP, -glIP,
                                    -glIP, -glIP, +glIP, +glIP, -glIP, +glIP,
                                    +glIP, +glIP, +glIP, -glIP, +glIP, +glIP};

  // standard integration location +/ sqrt(3)/3
  const double gIP = 0.577350269189626; // std::sqrt(3.0) / 3.0;
  const double intgLoc_[numIntPoints_ * nDim_] = {
    -gIP, -gIP, -gIP, +gIP, -gIP, -gIP, +gIP, +gIP, -gIP, -gIP, +gIP, -gIP,
    -gIP, -gIP, +gIP, +gIP, -gIP, +gIP, +gIP, +gIP, +gIP, -gIP, +gIP, +gIP};

  void
  hex8_fem_shape_fcn(const int numIp, const double* isoParCoord, double* shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void hex8_fem_shape_fcn(
    const int numIp,
    const double* isoParCoord,
    SharedMemView<SCALAR**, SHMEM> shpfc);

  void
  hex8_fem_derivative(const int npt, const double* par_coord, double* deriv);

  KOKKOS_FUNCTION void hex8_fem_derivative(
    const int npt,
    const double* par_coord,
    SharedMemView<DoubleType***, DeviceShmem> deriv);
};

} // namespace nalu
} // namespace sierra

#endif
