/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <master_element/MasterElementHO.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <NaluEnv.h>
#include <master_element/MasterElement.h>
#include <FORTRAN_Proto.h>

#include <BuildTemplates.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <element_promotion/ElementDescription.h>

namespace sierra{
namespace nalu{

void gradient_2d(
  int nodesPerElement,
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  double* POINTER_RESTRICT grad,
  double* POINTER_RESTRICT det_j)
{
  constexpr int dim = 2;

  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;

  //compute Jacobian
  for (int node = 0; node < nodesPerElement; ++node) {
    const int vector_offset = dim * node;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  *det_j = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;

  const double inv_det_j = 1.0 / (*det_j);

  const double ds1_dx =  inv_det_j*dy_ds2;
  const double ds2_dx = -inv_det_j*dy_ds1;

  const double ds1_dy = -inv_det_j*dx_ds2;
  const double ds2_dy =  inv_det_j*dx_ds1;

  for (int node = 0; node < nodesPerElement; ++node) {
    const int vector_offset = dim * node;

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy;
  }
}

void gradient_3d(
  int nodesPerElement,
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  double* POINTER_RESTRICT grad,
  double* POINTER_RESTRICT det_j)
{
  constexpr int dim = 3;

  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;

  //compute Jacobian
  int vector_offset = 0;
  for (int node = 0; node < nodesPerElement; ++node) {
    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double zCoord = elemNodalCoords[vector_offset + 2];

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;

    vector_offset += dim;
  }

  *det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
         + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
         + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  const double inv_det_j = 1.0 / (*det_j);

  const double ds1_dx = inv_det_j*(dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3);
  const double ds2_dx = inv_det_j*(dz_ds1 * dy_ds3 - dy_ds1 * dz_ds3);
  const double ds3_dx = inv_det_j*(dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2);

  const double ds1_dy = inv_det_j*(dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3);
  const double ds2_dy = inv_det_j*(dx_ds1 * dz_ds3 - dz_ds1 * dx_ds3);
  const double ds3_dy = inv_det_j*(dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2);

  const double ds1_dz = inv_det_j*(dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3);
  const double ds2_dz = inv_det_j*(dy_ds1 * dx_ds3 - dx_ds1 * dy_ds3);
  const double ds3_dz = inv_det_j*(dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);

  // metrics
  vector_offset = 0;
  for (int node = 0; node < nodesPerElement; ++node) {
    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx + dn_ds3 * ds3_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy + dn_ds3 * ds3_dy;
    grad[vector_offset + 2] = dn_ds1 * ds1_dz + dn_ds2 * ds2_dz + dn_ds3 * ds3_dz;

    vector_offset += dim;
  }
}


KOKKOS_FUNCTION
HigherOrderQuad3DSCS::HigherOrderQuad3DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
  nodeMap(make_node_map_quad(basis.order())),
  nodes1D_(basis.order()+1)
{
  surfaceDimension_ = 2;
  MasterElement::nDim_ = 3;
  nodesPerElement_ = nodes1D_*nodes1D_;

  // set up integration rule and relevant maps on scs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
}

void
HigherOrderQuad3DSCS::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (nodesPerElement_) * ( quadrature_.num_quad() * quadrature_.num_quad() );

  // define ip node mappings
  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int scalar_index = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      for (int j = 0; j < quadrature_.num_quad(); ++j) {
        for (int i = 0; i < quadrature_.num_quad(); ++i) {
          //integration point location
          intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(k,i);
          intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);

          //weight
          ipWeights_(scalar_index) = quadrature_.integration_point_weight(k,l,i,j);

          //sub-control volume association
          ipNodeMap_(scalar_index) = nodeMap(l,k);

          // increment indices
          ++scalar_index;
        }
      }
    }
  }
}

void
HigherOrderQuad3DSCS::shape_fcn(double* shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int *
HigherOrderQuad3DSCS::ipNodeMap(
  int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return &ipNodeMap_(0);
}

void
HigherOrderQuad3DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double * /*error*/)
{
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  Kokkos::View<double[3]> areaVector("areaVector");
  int grad_offset = 0;
  int grad_inc = surfaceDimension_ * nodesPerElement_;

  int vector_offset = 0;
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    //compute area vector for this ip
    area_vector( &coords[0], &shapeDerivs_.data()[grad_offset], areaVector );

    // apply quadrature weight and orientation (combined as weight)
    for (int j = 0; j < nDim_; ++j) {
      areav[vector_offset+j]  = ipWeights_(ip) * areaVector[j];
    }
    vector_offset += nDim_;
    grad_offset += grad_inc;
  }
}

void
HigherOrderQuad3DSCS::area_vector(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  Kokkos::View<double[3]> areaVector) const
{
  // return the normal area vector given shape derivatives dnds OR dndt
  double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
  double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const int surface_vector_offset = surfaceDimension_ * node;

    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDeriv[surface_vector_offset+0];
    const double dn_ds2 = shapeDeriv[surface_vector_offset+1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
  }

  //cross product
  areaVector(0) = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
  areaVector(1) = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
  areaVector(2) = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

KOKKOS_FUNCTION
HigherOrderQuad2DSCV::HigherOrderQuad2DSCV(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
  nodeMap(make_node_map_quad(basis.order())),
  nodes1D_(basis.order()+1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = nodes1D_*nodes1D_;

  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);

}

void
HigherOrderQuad2DSCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (nodesPerElement_) * ( quadrature_.num_quad() * quadrature_.num_quad() );

  // define ip node mappings
  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int scalar_index = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      for (int j = 0; j < quadrature_.num_quad(); ++j) {
        for (int i = 0; i < quadrature_.num_quad(); ++i) {
          intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(k,i);
          intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);
          ipWeights_(scalar_index) = quadrature_.integration_point_weight(k,l,i,j);
          ipNodeMap_(scalar_index) = nodeMap(l, k);

          // increment indices
          ++scalar_index;
        }
      }
    }
  }
}

void
HigherOrderQuad2DSCV::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int *
HigherOrderQuad2DSCV::ipNodeMap(int /*ordinal*/) const
{
  return &ipNodeMap_(0);
}

void
HigherOrderQuad2DSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    const double det_j = jacobian_determinant(coords, &shapeDerivs_.data()[grad_offset] );
    volume[ip] = ipWeights_(ip) * det_j;

    //flag error
    if (det_j < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

double
HigherOrderQuad2DSCV::jacobian_determinant(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = node * nDim_;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];

    const double dn_ds1  = shapeDerivs[vector_offset + 0];
    const double dn_ds2  = shapeDerivs[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  const double det_j = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
  return det_j;
}

void HigherOrderQuad2DSCV::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "Grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    for (int j = 0; j < grad_inc; ++j) {
      deriv[grad_offset + j] = shapeDerivs_.data()[grad_offset +j];
    }

    gradient_2d(
      nodesPerElement_,
      &coords[0],
      &shapeDerivs_.data()[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }
    grad_offset += grad_inc;
  }
}

KOKKOS_FUNCTION
HigherOrderQuad2DSCS::HigherOrderQuad2DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
  nodeMap(make_node_map_quad(basis.order())),
  faceNodeMap(make_face_node_map_quad(basis.order())),
  sideNodeOrdinals_(make_side_node_ordinal_map_quad(basis.order())),
  nodes1D_(basis.order()+1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = nodes1D_*nodes1D_;

  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = basis_.eval_deriv_weights(intgExpFace_);
}

double parametric_distance_quad(const double* x)
{
  double absXi  = std::abs(x[0]);
  double absEta = std::abs(x[1]);
  return (absXi > absEta) ? absXi : absEta;
}


double HigherOrderQuad2DSCS::isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord)
{
  std::array<double, 2> initialGuess = {{ 0.0, 0.0 }};
  int maxIter = 50;
  double tolerance = 1.0e-16;
  double deltaLimit = 1.0e4;

  bool converged = isoparameteric_coordinates_for_point_2d(
      basis_,
      elemNodalCoord,
      pointCoord,
      isoParCoord,
      initialGuess,
      maxIter,
      tolerance,
      deltaLimit
  );
  ThrowAssertMsg(parametric_distance_quad(isoParCoord) < 1.0 + 1.0e-6 || !converged,
      "Inconsistency in parametric distance calculation");

  return (converged) ? parametric_distance_quad(isoParCoord) : std::numeric_limits<double>::max();
}

void HigherOrderQuad2DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result)
{
  const auto& weights = basis_.point_interpolation_weights(isoParCoord);
  for (int n = 0; n < nComp; ++n) {
    result[n] = ddot(weights.data(), field + n * nodesPerElement_, nodesPerElement_);
  }
}

void
HigherOrderQuad2DSCS::set_interior_info()
{
  const int linesPerDirection = nodes1D_ - 1;
  const int ipsPerLine = quadrature_.num_quad() * nodes1D_;
  const int numLines = linesPerDirection * nDim_;

  numIntPoints_ = numLines * ipsPerLine;

  // define L/R mappings
  lrscv_ = Kokkos::View<int*>("lsrcv_", 2*numIntPoints_);

  // standard integration location
  intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  // specify integration point locations in a dimension-by-dimension manner

  //u-direction
  int lrscv_index = 0;
  int scalar_index = 0;
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {

      int leftNode;
      int rightNode;
      int orientation;
      if (m % 2 == 0) {
        leftNode  = nodeMap(m + 0, l);
        rightNode = nodeMap(m + 1, l);
        orientation = -1;
      }
      else {
        leftNode  = nodeMap(m + 1, l);
        rightNode = nodeMap(m + 0, l);
        orientation = +1;
      }

      for (int j = 0; j < quadrature_.num_quad(); ++j) {

        lrscv_(lrscv_index) = leftNode;
        lrscv_(lrscv_index + 1) = rightNode;

        intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(l,j);
        intgLoc_(scalar_index, 1) = quadrature_.scs_loc(m);

        //compute the quadrature weight
        ipWeights_(scalar_index) = orientation*quadrature_.integration_point_weight(l,j);

        ++scalar_index;
        lrscv_index += 2;
      }
    }
  }

  //t-direction
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {

      int leftNode;
      int rightNode;
      int orientation;
      if (m % 2 == 0) {
        leftNode  = nodeMap(l,m);
        rightNode = nodeMap(l,m+1);
        orientation = +1;
      }
      else {
        leftNode  = nodeMap(l,m+1);
        rightNode = nodeMap(l,m);
        orientation = -1;
      }

      for (int j = 0; j < quadrature_.num_quad(); ++j) {

        lrscv_(lrscv_index)   = leftNode;
        lrscv_(lrscv_index+1) = rightNode;

        intgLoc_(scalar_index, 0) = quadrature_.scs_loc(m);
        intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);

        //compute the quadrature weight
        ipWeights_(scalar_index) = orientation*quadrature_.integration_point_weight(l,j);

        ++scalar_index;
        lrscv_index += 2;
      }
    }
  }
}

void
HigherOrderQuad2DSCS::set_boundary_info()
{
  const int numFaces = 2*nDim_;
  const int nodesPerFace = nodes1D_;
  const int linesPerDirection = nodes1D_-1;
  ipsPerFace_ = nodesPerFace*quadrature_.num_quad();

  const int numFaceIps = numFaces*ipsPerFace_;

  oppFace_ =Kokkos::View<int*>("oppFace_", numFaceIps);
  ipNodeMap_ = Kokkos::View<int*>("ipNodeMap_",numFaceIps);
  oppNode_ = Kokkos::View<int*>("oppNode", numFaceIps);
  intgExpFace_=Kokkos::View<double**>("intgExpFace_",numFaceIps,nDim_);


  auto face_node_number = [&] (int number,int faceOrdinal)
  {
    return faceNodeMap(faceOrdinal,number);
  };

  const std::array<int, 4> faceToLine = {{
      0,
      2*linesPerDirection-1,
      linesPerDirection-1,
      linesPerDirection
  }};

  const std::array<double, 4> faceLoc = {{-1.0, +1.0, +1.0, -1.0}};

  int scalar_index = 0;
  int faceOrdinal = 0; //bottom face
  int oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = nodeMap(1, k);

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_(scalar_index)*nDim_, 0);
      intgExpFace_(scalar_index, 1) = faceLoc[faceOrdinal];

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 1; //right face
  oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = nodeMap(k, nodes1D_-2);

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0) = faceLoc[faceOrdinal];
      intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_(scalar_index)*nDim_, 1);

      ++scalar_index;
      ++oppFaceIndex;
    }
  }


  faceOrdinal = 2; //top face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  int elemNodeM1 = static_cast<int>(nodes1D_-1);
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = nodeMap(nodes1D_-2, k);
    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_(scalar_index)*nDim_, 0);
      intgExpFace_(scalar_index, 1) = faceLoc[faceOrdinal];

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 3; //left face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = nodeMap(k,1);
    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0)   = faceLoc[faceOrdinal];
      intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_(scalar_index)*nDim_, 1);

      ++scalar_index;
      ++oppFaceIndex;
    }
  }
}

void
HigherOrderQuad2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int *
HigherOrderQuad2DSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_(ordinal*ipsPerFace_);
}

const int *
HigherOrderQuad2DSCS::side_node_ordinals(int ordinal) const
{
  return &sideNodeOrdinals_(ordinal,0);
}

void
HigherOrderQuad2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  //returns the normal vector (dyds,-dxds) for constant t curves
  //returns the normal vector (dydt,-dxdt) for constant s curves
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  constexpr int dim = 2;
  int ipsPerDirection = numIntPoints_ / dim;

  int index = 0;

   //returns the normal vector x_u x x_s for constant t surfaces
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_(index,0,0), &areav[index*dim]);
    ++index;
  }

  //returns the normal vector x_t x x_u for constant s curves
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_(index,0,0), &areav[index*dim]);
    ++index;
  }

  // Multiply with the integration point weighting
  for (int ip = 0; ip < numIntPoints_; ++ip) {

    double weight = ipWeights_(ip);
    areav[ip * dim + 0] *= weight;
    areav[ip * dim + 1] *= weight;
  }
}

void HigherOrderQuad2DSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "Grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    for (int j = 0; j < grad_inc; ++j) {
      deriv[grad_offset + j] = shapeDerivs_.data()[grad_offset +j];
    }

    gradient_2d(
      nodesPerElement_,
      &coords[0],
      &shapeDerivs_.data()[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

void
HigherOrderQuad2DSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "face_grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  const int face_offset =  nDim_ * ipsPerFace_ * nodesPerElement_ * face_ordinal;
  const double* const faceShapeDerivs = &expFaceShapeDerivs_.data()[face_offset];

  for (int ip = 0; ip < ipsPerFace_; ++ip) {
    gradient_2d(
      nodesPerElement_,
      coords,
      &faceShapeDerivs[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
   );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}


const int *
HigherOrderQuad2DSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_.data();
}

int
HigherOrderQuad2DSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_(ordinal*ipsPerFace_+node);
}

int
HigherOrderQuad2DSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_(ordinal*ipsPerFace_+node);
}

template <Jacobian::Direction direction> void
HigherOrderQuad2DSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT normalVec ) const
{
  constexpr int s1Component = (direction == Jacobian::S_DIRECTION) ?
      Jacobian::T_DIRECTION : Jacobian::S_DIRECTION;

  double dxdr = 0.0;  double dydr = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[vector_offset+s1Component] * xCoord;
    dydr += shapeDeriv[vector_offset+s1Component] * yCoord;
  }

  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}

void HigherOrderQuad2DSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(twod_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}

KOKKOS_FUNCTION
HigherOrderEdge2DSCS::HigherOrderEdge2DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(basis),
  quadrature_(quadrature),
  nodeMap(make_node_map_edge(basis.order())),
  nodes1D_(basis.order()+1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = nodes1D_;
  numIntPoints_ = quadrature_.num_quad() * nodes1D_;

  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ =  Kokkos::View<double*[1]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  int scalar_index = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    for (int i = 0; i < quadrature_.num_quad(); ++i) {
      // TODO(psakiev) double check this
      intgLoc_(scalar_index, 0)  = quadrature_.integration_point_location(k,i);
      ipWeights_(scalar_index) = quadrature_.integration_point_weight(k,i);
      ipNodeMap_(scalar_index) = nodeMap(k);
      ++scalar_index;
    }
  }
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
}

const int *
HigherOrderEdge2DSCS::ipNodeMap(int /*ordinal*/) const
{
  return ipNodeMap_.data();
}

void
HigherOrderEdge2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  Kokkos::View<double[2]> areaVector("areaVector");
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  int grad_offset = 0;
  const int grad_inc = nodesPerElement_;

  int vec_offset = 0;
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    // calculate the area vector
    area_vector( &coords[0],
      &shapeDerivs_.data()[grad_offset],
      areaVector );

    // weight the area vector with the Gauss-quadrature weight for this IP
    areav[vec_offset + 0] = ipWeights_(ip) * areaVector(0);
    areav[vec_offset + 1] = ipWeights_(ip) * areaVector(1);

    grad_offset += grad_inc;
    vec_offset += nDim_;
  }
}

void
HigherOrderEdge2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
   for (int j = 0; j < numShape; ++j) {
     shpfc[j] = shapeFunctionVals_.data()[j];
   }
}

void
HigherOrderEdge2DSCS::area_vector(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  Kokkos::View<double[2]> areaVector) const
{
  double dxdr = 0.0;  double dydr = 0.0;
  int vector_offset = 0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[node] * xCoord;
    dydr += shapeDeriv[node] * yCoord;

    vector_offset += nDim_;
  }
  areaVector(0) =  dydr;
  areaVector(1) = -dxdr;
}

}  // namespace nalu
} // namespace sierra
