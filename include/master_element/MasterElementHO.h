/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MasterElementHO_h
#define MasterElementHO_h

#include <master_element/MasterElement.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/NodeMapMaker.h>



#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <array>

namespace sierra{
namespace nalu{

class LagrangeBasis;
class TensorProductQuadratureRule;

// 3D Quad 9
class HigherOrderQuad3DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  HigherOrderQuad3DSCS(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderQuad3DSCS() {}

  void shape_fcn(double *shpfc) final;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();
  void eval_shape_functions_at_ips();
  void eval_shape_derivs_at_ips();

  void area_vector(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    Kokkos::View<double[3]> areaVector) const;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const int nodes1D_;

  Kokkos::View<double**>  shapeFunctionVals_;
  Kokkos::View<double***>  shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
  int surfaceDimension_;
};

class HigherOrderQuad2DSCV final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::grad_op;

  KOKKOS_FUNCTION
  HigherOrderQuad2DSCV(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderQuad2DSCV() {}

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



  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }


  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();

  double jacobian_determinant(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDerivs ) const;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const int nodes1D_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***>  shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
};
class HigherOrderQuad2DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::grad_op;
  using MasterElement::face_grad_op;
  using MasterElement::gij;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  HigherOrderQuad2DSCS(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderQuad2DSCS() {}

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

  int opposingNodes(
    const int ordinal, const int node) final;

  int opposingFace(
    const int ordinal, const int node) final;

  const int * side_node_ordinals(int ordinal = 0) const final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  Kokkos::View<int*> lrscv_;

  void set_interior_info();
  void set_boundary_info();

  template <Jacobian::Direction direction> void
  area_vector(
    const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT normalVec ) const;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const Kokkos::View<int**> faceNodeMap;
  const Kokkos::View<int**> sideNodeOrdinals_;
  const int nodes1D_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***>  shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
  int ipsPerFace_;
  Kokkos::View<double***> expFaceShapeDerivs_;
  Kokkos::View<int*> oppNode_;
  Kokkos::View<int*> oppFace_;
  Kokkos::View<double**> intgExpFace_;
};

class HigherOrderEdge2DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  explicit HigherOrderEdge2DSCS(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderEdge2DSCS() = default;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) final;

  void shape_fcn(
    double *shpfc) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

private:
  void area_vector(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    Kokkos::View<double[2]> areaVector) const;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int*> nodeMap;
  const int nodes1D_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***> shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
};

} // namespace nalu
} // namespace Sierra

#endif
