/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef Hex27CVFEM_h
#define Hex27CVFEM_h

#include <master_element/MasterElement.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/MasterElementFunctions.h>

#include <SimdInterface.h>
#include <Kokkos_Core.hpp>
#include <AlgTraits.h>

#include <stk_util/util/ReportHandler.hpp>

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <type_traits>

namespace sierra{
namespace nalu{

class HexahedralP2Element : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex27;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  HexahedralP2Element();
  virtual ~HexahedralP2Element() {}

  void shape_fcn(double *shpfc);
  void shifted_shape_fcn(double *shpfc);


  template <typename ViewType>
  ViewType copy_interpolation_weights_to_view(const std::vector<double>& interps)
  {
    ViewType interpWeights{"interpolation_weights"};

    int shape_index = 0;
    for (unsigned ip = 0; ip < interpWeights.extent(0); ++ip) {
      for (unsigned n = 0; n < 27; ++n) {
        interpWeights(ip, n) = interps[shape_index];
        ++shape_index;
      }
    }
    return interpWeights;
  }



  template <typename ViewType>
  ViewType copy_deriv_weights_to_view(const std::vector<double>& derivs)
  {
    ViewType referenceGradWeights{"reference_gradient_weights"};

    int deriv_index = 0;
    for (unsigned ip = 0; ip < referenceGradWeights.extent(0); ++ip) {
      for (unsigned n = 0; n < 27; ++n) {
        for (unsigned d = 0; d < 3; ++d) {
          referenceGradWeights(ip,n,d) = derivs[deriv_index];
          ++deriv_index;
        }
      }
    }
    return referenceGradWeights;
  }

  template <typename ViewType>
  ViewType copy_interpolation_weights_to_view()
  {
    ViewType interpWeights{"interpolation_weights"};

    int shape_index = 0;
    for (unsigned ip = 0; ip < interpWeights.extent(0); ++ip) {
      for (unsigned n = 0; n < 27; ++n) {
        interpWeights(ip, n) = shapeFunctions_[shape_index];
        ++shape_index;
      }
    }
    return interpWeights;
  }

  template <typename ViewType>
  ViewType copy_deriv_weights_to_view()
  {
    ViewType referenceGradWeights{"reference_gradient_weights"};

    int deriv_index = 0;
    for (unsigned ip = 0; ip < referenceGradWeights.extent(0); ++ip) {
      for (unsigned n = 0; n < 27; ++n) {
        for (unsigned d = 0; d < 3; ++d) {
          referenceGradWeights(ip,n,d) = shapeDerivs_[deriv_index];
          ++deriv_index;
        }
      }
    }
    return referenceGradWeights;
  }




protected:
  struct ContourData {
    Jacobian::Direction direction;
    double weight;
  };

  int tensor_product_node_map(int i, int j, int k) const;

  double gauss_point_location(
    int nodeOrdinal,
    int gaussPointOrdinal) const;

  double shifted_gauss_point_location(
    int nodeOrdinal,
    int gaussPointOrdinal) const;

  double tensor_product_weight(
    int s1Node, int s2Node, int s3Node,
    int s1Ip, int s2Ip, int s3Ip) const;

  double tensor_product_weight(
    int s1Node, int s2Node,
    int s1Ip, int s2Ip) const;

  virtual void eval_shape_functions_at_ips();
  virtual void eval_shape_functions_at_shifted_ips();

  virtual void eval_shape_derivs_at_ips();
  virtual void eval_shape_derivs_at_shifted_ips();

  void eval_shape_derivs_at_face_ips();

  void set_quadrature_rule();
  void GLLGLL_quadrature_weights();

  void hex27_shape_deriv(
    int npts,
    const double *par_coord,
    double* shape_fcn
  ) const;

  double parametric_distance(const std::array<double, 3>& x);

  virtual void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  virtual double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord);

  const double scsDist_ = std::sqrt(3.0)/3.0;
  const int nodes1D_    = 3;
  const int numQuad_    = 2;

  // quadrature info
  const double gaussAbscissae_[2] = {-std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0};
  const double gaussWeight_[2]    = {0.5, 0.5};
  const double gaussAbscissaeShift_[6] = {-1.0, -1.0, 0.0, 0.0,  1.0, 1.0};

  const double scsEndLoc_[4] =  { -1.0, -scsDist_, scsDist_, 1.0 };
 
  // map the standard stk node numbering to a tensor-product style node numbering (i.e. node (m,l,k) -> m+npe*l+npe^2*k)
  const int    stkNodeMap_[3][3][3] = {
                {{ 0,  8,  1}, // bottom front edge
                 {11, 21,  9}, // bottom mid-front edge
                 { 3, 10,  2}},// bottom back edge
                {{12, 25, 13}, // mid-top front edge
                 {23, 20, 24}, // mid-top mid-front edge
                 {15, 26, 14}},// mid-top back edge
                {{ 4, 16,  5}, // top front edge
                 {19, 22, 17}, // top mid-front edge
                 { 7, 18,  6}} // top back edge
                };

  const int  sideNodeOrdinals_[6][9] = {
      {0, 1, 5, 4, 8,13,16,12,25}, //ordinal 0
      {1, 2, 6, 5, 9,14,17,13,24}, //ordinal 1
      {2, 3, 7, 6,10,15,18,14,26}, //ordinal 2
      {0, 4, 7, 3,12,19,15,11,23}, //ordinal 3
      {0, 3, 2, 1,11,10, 9, 8,21}, //ordinal 4
      {4, 5, 6, 7,16,17,18,19,22}  //ordinal 5
  };

  std::vector<double> shapeFunctions_;
  std::vector<double> shapeFunctionsShift_;
  std::vector<double> shapeDerivs_;
  std::vector<double> shapeDerivsShift_;
  std::vector<double> expFaceShapeDerivs_;
private:
  void hex27_shape_fcn(
    int npts,
    const double *par_coord,
    double* shape_fcn
  ) const;
};

// 3D Quad 27 subcontrol volume
class Hex27SCV : public HexahedralP2Element
{
  using InterpWeightType = AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]>;
  using GradWeightType = AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_][AlgTraits::nDim_]>;

public:
  Hex27SCV();
  virtual ~Hex27SCV() {}

  const int * ipNodeMap(int ordinal = 0);

  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;

  template<typename ViewType> KOKKOS_FUNCTION void shape_fcn(ViewType &shpfc);
  void shape_fcn(SharedMemView<DoubleType**> &shpfc) final;
  void shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc) final;
  void determinant(SharedMemView<DoubleType**>& coords, SharedMemView<DoubleType*>& volume) final;

  void grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv);

  void Mij(
    const double *coords,
    double *metric,
    double *deriv);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  const InterpWeightType& shape_function_values()
  { return interpWeights_; }

  const GradWeightType& shape_function_derivatives()
  { return referenceGradWeights_; }

  template <typename GradViewType, typename CoordViewType, typename OutputViewType>
  void weighted_volumes(GradViewType referenceGradWeights, CoordViewType coords, OutputViewType volume)
  {
    generic_determinant_3d<AlgTraits>(referenceGradWeights, coords, volume);
    for (int ip = 0 ; ip < AlgTraits::numScvIp_; ++ip) {
      volume(ip) *= ipWeight_[ip];
    }
  }


private:
  void set_interior_info();

  double jacobian_determinant(
    const double *POINTER_RESTRICT elemNodalCoords,
    const double *POINTER_RESTRICT shapeDerivs ) const;

  InterpWeightType interpWeights_;
  GradWeightType referenceGradWeights_;

  InterpWeightType shiftedInterpWeights_;
  GradWeightType shiftedReferenceGradWeights_;

  std::vector<double> ipWeight_;
};

// 3D Hex 27 subcontrol surface
class Hex27SCS : public HexahedralP2Element
{
  using InterpWeightType = AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]>;
  using GradWeightType = AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_][AlgTraits::nDim_]>;
  using ExpGradWeightType = AlignedViewType<DoubleType[6*AlgTraitsQuad9Hex27::numFaceIp_][AlgTraits::nodesPerElement_][AlgTraits::nDim_]>;

public:
  Hex27SCS();
  virtual ~Hex27SCS() {}

  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::determinant;

  template<typename ViewType> KOKKOS_FUNCTION void shape_fcn(ViewType &shpfc);
  void shape_fcn(SharedMemView<DoubleType**> &shpfc);
  void shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc);

  template<typename ViewTypeCoord, typename ViewTypeGrad>
  KOKKOS_FUNCTION
  void grad_op(
    ViewTypeCoord& coords,
    ViewTypeGrad&  gradop,
    ViewTypeGrad&  deriv);
 

  void grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void determinant(SharedMemView<DoubleType**>&coords,  SharedMemView<DoubleType**>&areav);

  void gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv);

  void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv);

  void Mij(
    const double *coords,
    double *metric,
    double *deriv);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

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

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv);

  void general_face_grad_op(
    const int face_ordinal,
    const double *isoParCoord,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void sidePcoords_to_elemPcoords(
    const int & side_ordinal,
    const int & npoints,
    const double *side_pcoords,
    double *elem_pcoords);

  const int * adjacentNodes();

  const int * ipNodeMap(int ordinal = 0);

  int opposingNodes(
    const int ordinal, const int node);

  int opposingFace(
    const int ordinal, const int node);

  const int* side_node_ordinals(int sideOrdinal) final;
  using MasterElement::side_node_ordinals;

  const InterpWeightType& shape_function_values()
  { return interpWeights_; }

  const GradWeightType& shape_function_derivatives()
  { return referenceGradWeights_; }

  template <typename GradViewType, typename CoordViewType, typename OutputViewType>
  void weighted_area_vectors(GradViewType referenceGradWeights, CoordViewType coords, OutputViewType areav)
  {
    using ftype = typename CoordViewType::value_type;

    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value, "Incompatiable value type for views");
    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 2, "area_vector view assumed to be 2D");

    static_assert (AlgTraits::numScsIp_ % AlgTraits::nDim_ == 0, "Number of ips incorrect");
    constexpr int ipsPerDirection = AlgTraits::numScsIp_ / AlgTraits::nDim_;
    constexpr int t_start = 1*ipsPerDirection;
    constexpr int s_start = 2*ipsPerDirection;

    // this relies on the ips being laid out direction-by-direction,
    // specifically in the U->T->S order
    for (int ip = 0; ip < t_start; ++ip) {
      ThrowAssert(ipInfo_[ip].direction == Jacobian::U_DIRECTION);
      area_vector<Jacobian::U_DIRECTION>(ip, referenceGradWeights, coords, areav);
    }

    for (int ip = t_start; ip < s_start; ++ip) {
      ThrowAssert(ipInfo_[ip].direction == Jacobian::T_DIRECTION);
      area_vector<Jacobian::T_DIRECTION>(ip, referenceGradWeights, coords, areav);
    }

    for (int ip = s_start; ip < AlgTraits::numScsIp_; ++ip) {
      ThrowAssert(ipInfo_[ip].direction == Jacobian::S_DIRECTION);
      area_vector<Jacobian::S_DIRECTION>(ip, referenceGradWeights, coords, areav);
    }

    for (int ip = 0; ip < 216; ++ip) {
      const ftype weight = ipInfo_[ip].weight;
      areav(ip, 0) *= weight;
      areav(ip, 1) *= weight;
      areav(ip, 2) *= weight;
    }
  }

protected:
  std::vector<ContourData> ipInfo_;

private:
  void set_interior_info();
  void set_boundary_info();

  template <Jacobian::Direction dir>
  void area_vector(const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT areaVector ) const;

  void gradient(
    const double *POINTER_RESTRICT elemNodalCoords,
    const double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT grad,
    double *POINTER_RESTRICT det_j ) const;

  template <int direction, typename GradViewType, typename CoordViewType, typename OutputViewType>
  void area_vector(int ip, GradViewType referenceGradWeights, CoordViewType coords, OutputViewType areav)
  {
    constexpr int s1Component = (direction == Jacobian::T_DIRECTION) ? Jacobian::S_DIRECTION : Jacobian::T_DIRECTION;
    constexpr int s2Component = (direction == Jacobian::U_DIRECTION) ? Jacobian::S_DIRECTION : Jacobian::U_DIRECTION;

    using ftype = typename CoordViewType::value_type;

    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value,
      "Incompatiable value type for views");

    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 2, "areav view assumed to be 2D");

    ftype sjac[3][2] = { {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0} };
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      const ftype dn_ds1 = referenceGradWeights(ip, n, s1Component);
      const ftype dn_ds2 = referenceGradWeights(ip, n, s2Component);

      sjac[0][0] += dn_ds1 * coords(n,0);
      sjac[0][1] += dn_ds2 * coords(n,0);

      sjac[1][0] += dn_ds1 * coords(n,1);
      sjac[1][1] += dn_ds2 * coords(n,1);

      sjac[2][0] += dn_ds1 * coords(n,2);
      sjac[2][1] += dn_ds2 * coords(n,2);
    }
    areav(ip, 0) = sjac[1][0] * sjac[2][1] - sjac[2][0] * sjac[1][1];
    areav(ip, 1) = sjac[2][0] * sjac[0][1] - sjac[0][0] * sjac[2][1];
    areav(ip, 2) = sjac[0][0] * sjac[1][1] - sjac[1][0] * sjac[0][1];
  }

  InterpWeightType interpWeights_;
  GradWeightType referenceGradWeights_;

  InterpWeightType shiftedInterpWeights_;
  GradWeightType shiftedReferenceGradWeights_;

  ExpGradWeightType expReferenceGradWeights_;

  int ipsPerFace_;
};

template<typename ViewType> 
KOKKOS_FUNCTION void Hex27SCV::shape_fcn(ViewType &shpfc)
{
  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      shpfc(ip,n) = interpWeights_(ip,n);
    }
  }
}
template<typename ViewType> 
KOKKOS_FUNCTION void Hex27SCS::shape_fcn(ViewType &shpfc)
{
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      shpfc(ip,n) = interpWeights_(ip,n);
    }
  }
}


template<typename ViewTypeCoord, typename ViewTypeGrad>
KOKKOS_FUNCTION void Hex27SCS::grad_op(
  ViewTypeCoord& coords,
  ViewTypeGrad&  gradop,
  ViewTypeGrad&  deriv)
{
  generic_grad_op<AlgTraits>(referenceGradWeights_, coords, gradop);

  // copy derivs as well.  These aren't used, but are part of the interface
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        deriv(ip,n,d) = referenceGradWeights_(ip,n,d);
      }
    }
  }
}


} // namespace nalu
} // namespace Sierra

#endif
