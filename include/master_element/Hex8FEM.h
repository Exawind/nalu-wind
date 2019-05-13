/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Hex8FEM_h
#define Hex8FEM_h

#include<master_element/MasterElement.h>

namespace sierra{
namespace nalu{

// Hex 8 FEM; -1.0 : 1.0 range
class Hex8FEM : public MasterElement
{
public:

  KOKKOS_FUNCTION
  Hex8FEM();
  KOKKOS_FUNCTION
  virtual ~Hex8FEM() = default;

  using AlgTraits = AlgTraitsHex8;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;
  using MasterElement::face_grad_op;
  using MasterElement::gij;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void shifted_grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  KOKKOS_FUNCTION void grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>&coords,
    SharedMemView<DoubleType***, DeviceShmem>&gradop,
    SharedMemView<DoubleType***, DeviceShmem>&deriv,
    SharedMemView<DoubleType*, DeviceShmem>&det_j) final;

  KOKKOS_FUNCTION void shifted_grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>&coords,
    SharedMemView<DoubleType***, DeviceShmem>&gradop,
    SharedMemView<DoubleType***, DeviceShmem>&deriv,
    SharedMemView<DoubleType*, DeviceShmem>&det_j) final;

  KOKKOS_FUNCTION void shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &shpfc) final;

  KOKKOS_FUNCTION void shifted_shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &shpfc) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double *error);

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc);

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv);

  virtual const double* integration_locations() const final {
    return intgLoc_;
  }
  virtual const double* integration_location_shift() const final {
    return intgLocShift_;
  }

  // weights; -1:1
  double weights_[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

private:
  static const int nDim_ = AlgTraits::nDim_;
  static const int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static const int numIntPoints_ = AlgTraits::numScvIp_;

  // shifted to nodes (Gauss Lobatto)
  const double glIP = 1.0;
  const double intgLocShift_[24] = {
   -glIP,  -glIP,  -glIP, 
   +glIP,  -glIP,  -glIP, 
   +glIP,  +glIP,  -glIP,
   -glIP,  +glIP,  -glIP,
   -glIP,  -glIP,  +glIP,
   +glIP,  -glIP,  +glIP,
   +glIP,  +glIP,  +glIP,
   -glIP,  +glIP,  +glIP};

  // standard integration location +/ sqrt(3)/3
  const double gIP = std::sqrt(3.0)/3.0;
  const double  intgLoc_[numIntPoints_*nDim_] = {
   -gIP,  -gIP,  -gIP, 
   +gIP,  -gIP,  -gIP, 
   +gIP,  +gIP,  -gIP,
   -gIP,  +gIP,  -gIP,
   -gIP,  -gIP,  +gIP,
   +gIP,  -gIP,  +gIP,
   +gIP,  +gIP,  +gIP,
   -gIP,  +gIP,  +gIP};


  double intgExpFace_[72];
  void hex8_fem_shape_fcn(
    const int  numIp,
    const double *isoParCoord,
    double *shpfc);

  KOKKOS_FUNCTION void hex8_fem_shape_fcn(
    const int  numIp,
    const double *isoParCoord,
    SharedMemView<DoubleType**, DeviceShmem> shpfc);

  void hex8_fem_derivative(
    const int npt, const double* par_coord,
    double* deriv);

  KOKKOS_FUNCTION void hex8_fem_derivative(
    const int npt, const double* par_coord,
    SharedMemView<DoubleType***, DeviceShmem> deriv);
};
    
} // namespace nalu
} // namespace Sierra

#endif
