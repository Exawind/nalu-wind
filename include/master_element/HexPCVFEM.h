/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HexPCVFEM_h
#define HexPCVFEM_h

#include <master_element/MasterElement.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/QuadNElementDescription.h>

#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <vector>
#include <array>

namespace sierra{
namespace nalu{


struct ElementDescription;

class LagrangeBasis;
class TensorProductQuadratureRule;

class HigherOrderHexSCV final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  HigherOrderHexSCV(LagrangeBasis basis, TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderHexSCV() {}

  void shape_fcn(double *shpfc) final;
  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error ) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

private:
  void set_interior_info();

  double jacobian_determinant(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDerivs ) const;

  const int nodes1D_;
  const Kokkos::View<int***> nodeMap;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***> shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
};

// 3D Hex 27 subcontrol surface
class HigherOrderHexSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::gij;
  using MasterElement::face_grad_op;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  HigherOrderHexSCS(LagrangeBasis basis, TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderHexSCS() {}

  void shape_fcn(double *shpfc) final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error) final;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) final;

  double isInElement(
      const double *elemNodalCoord,
      const double *pointCoord,
      double *isoParCoord) final;

  void interpolatePoint(
      const int &nComp,
      const double *isoParCoord,
      const double *field,
      double *result) final;

  KOKKOS_FUNCTION const int * adjacentNodes() final;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  const int * side_node_ordinals(int ordinal = 0) const final;

  int opposingNodes(
    const int ordinal, const int node) final;

  int opposingFace(
    const int ordinal, const int node) final;

  KOKKOS_FUNCTION void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

private:
  void set_interior_info();
  void set_boundary_info();

  int opposing_face_map(int k, int l, int i, int j, int face_index);

  template <Jacobian::Direction direction> void
  area_vector(
    const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT areaVector) const;

  const int nodes1D_;
  const int numQuad_;
  const int ipsPerFace_;

  const Kokkos::View<int***> nodeMap;
  const Kokkos::View<int***> faceNodeMap;
  const Kokkos::View<int**> sideNodeOrdinals_;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  Kokkos::View<int**> lrscv_;
  Kokkos::View<int*> oppNode_;
  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***> shapeDerivs_;
  Kokkos::View<double***> expFaceShapeDerivs_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<double**> intgExpFace_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<int*> ipNodeMap_;
  Kokkos::View<int*> oppFace_;

  AlignedViewType<DoubleType**[3]> expRefGradWeights_;
};

} // namespace nalu
} // namespace Sierra

#endif
