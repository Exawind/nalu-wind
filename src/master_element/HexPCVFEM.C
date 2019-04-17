/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <master_element/HexPCVFEM.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/NodeMapMaker.h>
#include <element_promotion/TensorProductQuadratureRule.h>

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


namespace sierra{
namespace nalu{

namespace {
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
}

KOKKOS_FUNCTION
HigherOrderHexSCV::HigherOrderHexSCV(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
  : MasterElement(),
    nodes1D_(basis.order() + 1),
    nodeMap(make_node_map_hex(basis.order(), true)),
    basis_(std::move(basis)),
    quadrature_(std::move(quadrature))
{
  MasterElement::nDim_ = 3;
  MasterElement::nodesPerElement_ = nodes1D_ * nodes1D_ * nodes1D_;
  MasterElement::numIntPoints_ = nodesPerElement_ * (quadrature_.num_quad() * quadrature_.num_quad() * quadrature_.num_quad());

  ipNodeMap_ = Kokkos::View<int*>("ip_node_map", numIntPoints_);
  intgLoc_ = Kokkos::View<double**>("integration_point_location", numIntPoints_, 3);
  ipWeights_ = Kokkos::View<double*>("integration_point_weights", numIntPoints_);

  int flat_index = 0;
  for (int n = 0; n < nodes1D_; ++n) {
    for (int m = 0; m < nodes1D_; ++m) {
      for (int l = 0; l < nodes1D_; ++l) {
        for (int k = 0; k < quadrature_.num_quad(); ++k) {
          for (int j = 0; j < quadrature_.num_quad(); ++j) {
            for (int i = 0; i < quadrature_.num_quad(); ++i) {
              intgLoc_(flat_index, 0)= quadrature_.integration_point_location(l,i);
              intgLoc_(flat_index, 1) = quadrature_.integration_point_location(m,j);
              intgLoc_(flat_index, 2) = quadrature_.integration_point_location(n,k);
              ipWeights_[flat_index] = quadrature_.integration_point_weight(l, m, n, i, j, k);
              ipNodeMap_[flat_index] = nodeMap(n, m, l);
              ++flat_index;
            }
          }
        }
      }
    }
  }
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
}

void
HigherOrderHexSCV::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int* HigherOrderHexSCV::ipNodeMap(int) const { return ipNodeMap_.data(); }

void HigherOrderHexSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  int grad_offset = 0;
  const int grad_inc = nDim_ * nodesPerElement_;

  for (int ip = 0; ip < numIntPoints_; ++ip, grad_offset += grad_inc) {
    const double det_j = jacobian_determinant(coords,  &shapeDerivs_.data()[grad_offset]);
    volume[ip] = ipWeights_[ip] * det_j;

    if (det_j < tiny_positive_value()) {
      *error = 1.0;
    }
  }
}

double HigherOrderHexSCV::jacobian_determinant(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDerivs[vector_offset+0];
    const double dn_ds2 = shapeDerivs[vector_offset+1];
    const double dn_ds3 = shapeDerivs[vector_offset+2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;
  }

  const double det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
                     + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
                     + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  return det_j;
}

void HigherOrderHexSCV::grad_op(
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

    gradient_3d(nodesPerElement_, coords, &shapeDerivs_.data()[grad_offset], &gradop[grad_offset], &det_j[ip]);

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}


int ip_per_face(const TensorProductQuadratureRule& quad, const LagrangeBasis& basis) {
  return quad.num_quad() * quad.num_quad() * (basis.order() + 1)*(basis.order() + 1);
}

KOKKOS_FUNCTION
HigherOrderHexSCS::HigherOrderHexSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  nodes1D_(basis.order() + 1),
  numQuad_(quadrature.num_quad()),
  ipsPerFace_(nodes1D_ * nodes1D_ * numQuad_ * numQuad_),
  nodeMap(make_node_map_hex(basis.order(), true)),
  faceNodeMap(make_face_node_map_hex(basis.order())),
  sideNodeOrdinals_(make_side_node_ordinal_map_hex(basis.order())),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
  expRefGradWeights_("reference_gradient_weights", 6*ip_per_face(quadrature, basis), basis.num_nodes())
{
  MasterElement::nDim_ = 3;
  nodesPerElement_ = nodes1D_ * nodes1D_ * nodes1D_;
  numIntPoints_ = 3 * (nodes1D_ - 1) * ipsPerFace_;

  // set up integration rule and relevant maps on scs
  set_interior_info();

  // set up integration rule and relevant maps on faces
  set_boundary_info();

  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = basis_.eval_deriv_weights(intgExpFace_);
}

void
HigherOrderHexSCS::set_interior_info()
{
  const int surfacesPerDirection = nodes1D_ - 1;

  lrscv_ = Kokkos::View<int**>("left_right_state_mapping", numIntPoints_, 2);
  intgLoc_ = Kokkos::View<double**>("integration_point_location", numIntPoints_, 3);
  ipWeights_ = Kokkos::View<double*>("ip_weight", numIntPoints_);

  // specify integration point locations in a dimension-by-dimension manner
  // u direction: bottom-top (0-1)
  int scalar_index = 0;
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode; int orientation;
        if (m % 2 == 0) {
          leftNode = nodeMap(m, l, k);
          rightNode = nodeMap(m + 1, l, k);
          orientation = -1;
        }
        else {
          leftNode = nodeMap(m + 1, l, k);
          rightNode = nodeMap(m, l, k);
          orientation = +1;
        }

        for (int j = 0; j < quadrature_.num_quad(); ++j) {
          for (int i = 0; i < quadrature_.num_quad(); ++i) {
            lrscv_(scalar_index, 0) = leftNode;
            lrscv_(scalar_index, 1) = rightNode;

            intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(k,i);
            intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);
            intgLoc_(scalar_index, 2) = quadrature_.scs_loc(m);

            ipWeights_[scalar_index] = orientation * quadrature_.integration_point_weight(k, l, i, j);

            ++scalar_index;
          }
        }
      }
    }
  }

  // t direction: front-back (2-3)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode; int orientation;
        if (m % 2 == 0) {
          leftNode = nodeMap(l, m + 0, k);
          rightNode = nodeMap(l, m + 1, k);
          orientation = -1;
        }
        else {
          leftNode = nodeMap(l, m + 1, k);
          rightNode = nodeMap(l, m + 0, k);
          orientation = +1;
        }

        for (int j = 0; j < quadrature_.num_quad(); ++j) {
          for (int i = 0; i < quadrature_.num_quad(); ++i) {
            lrscv_(scalar_index, 0)     = leftNode;
            lrscv_(scalar_index, 1) = rightNode;

            intgLoc_(scalar_index, 0)    = quadrature_.integration_point_location(k,i);
            intgLoc_(scalar_index, 1) = quadrature_.scs_loc(m);
            intgLoc_(scalar_index, 2) = quadrature_.integration_point_location(l,j);

            ipWeights_[scalar_index] = orientation * quadrature_.integration_point_weight(k, l, i, j);

            ++scalar_index;
          }
        }
      }
    }
  }

  //s direction: left-right (4-5)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode; int orientation;
        if (m % 2 == 0) {
          leftNode = nodeMap(l, k, m + 0);
          rightNode = nodeMap(l, k, m + 1);
          orientation = +1;
        }
        else {
          leftNode = nodeMap(l, k, m + 1);
          rightNode = nodeMap(l, k, m + 0);
          orientation = -1;
        }

        for (int j = 0; j < quadrature_.num_quad(); ++j) {
          for (int i = 0; i < quadrature_.num_quad(); ++i) {
            lrscv_(scalar_index, 0) = leftNode;
            lrscv_(scalar_index, 1) = rightNode;

            intgLoc_(scalar_index, 0) = quadrature_.scs_loc(m);
            intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(k,i);
            intgLoc_(scalar_index, 2) = quadrature_.integration_point_location(l,j);

            ipWeights_[scalar_index] = orientation * quadrature_.integration_point_weight(k, l, i, j);

            ++scalar_index;
          }
        }
      }
    }
  }
}

int HigherOrderHexSCS::opposing_face_map(int k, int l, int i, int j, int face_index)
{
  const int surfacesPerDirection = nodes1D_ - 1;
  const int faceToSurface[6] = {
      surfacesPerDirection,     // nearest scs face to t=-1.0
      3*surfacesPerDirection-1, // nearest scs face to s=+1.0, the last face
      2*surfacesPerDirection-1, // nearest scs face to t=+1.0
      2*surfacesPerDirection,   // nearest scs face to s=-1.0
      0,                        // nearest scs face to u=-1.0, the first face
      surfacesPerDirection-1    // nearest scs face to u=+1.0, the first face
  };

  const int face_offset = faceToSurface[face_index] * ipsPerFace_;
  const int node_index = k + nodes1D_ * l;
  const int node_offset = node_index * (numQuad_ * numQuad_);
  const int ip_index = face_offset + node_offset + i + numQuad_ * j;

  return ip_index;
}

void
HigherOrderHexSCS::set_boundary_info()
{
  const int numFaceIps = 6 * ipsPerFace_;

  oppFace_ = Kokkos::View<int*>("opposing_face_for_ip", numFaceIps);
  ipNodeMap_ = Kokkos::View<int*>("owning_node_for_ip", numFaceIps);
  oppNode_ = Kokkos::View<int*>("opposing_node_for_ip", numFaceIps);
  intgExpFace_ = Kokkos::View<double**>("exposed_face_integration_loc", numFaceIps, 3);

  // tensor-product style access to the map
  auto face_node_number = [&] (int i, int j, int faceOrdinal)
  {
    return faceNodeMap(faceOrdinal, j, i);
  };


  // location of the faces in the correct order
  const std::vector<double> faceLoc = {-1.0, +1.0, +1.0, -1.0, -1.0, +1.0};

  // Set points face-by-face
  int scalar_index = 0; int faceOrdinal = 0;

  // front face: t = -1.0: counter-clockwise
  faceOrdinal = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      const int oppNode = nodeMap(l,1,k);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opposing_face_map(k,l,i,j,faceOrdinal);

          intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_[scalar_index], 0);
          intgExpFace_(scalar_index, 1) = faceLoc[faceOrdinal];
          intgExpFace_(scalar_index, 2) = intgLoc_(oppFace_[scalar_index], 2);

          ++scalar_index;
        }
      }
    }
  }

  // right face: s = +1.0: counter-clockwise
  faceOrdinal = 1;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      const int oppNode = nodeMap(l,k,nodes1D_-2);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opposing_face_map(k,l,i,j,faceOrdinal);

          intgExpFace_(scalar_index, 0) = faceLoc[faceOrdinal];
          intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_[scalar_index], 1);
          intgExpFace_(scalar_index, 2) = intgLoc_(oppFace_[scalar_index], 2);

          ++scalar_index;
        }
      }
    }
  }

  // back face: t = +1.0: s-direction reversed
  faceOrdinal = 2;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = nodes1D_-1; k >= 0; --k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      const int oppNode = nodeMap(l,nodes1D_-2,k);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = numQuad_-1; i >= 0; --i) {
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opposing_face_map(k,l,i,j,faceOrdinal);

          intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_[scalar_index], 0);
          intgExpFace_(scalar_index, 1) = faceLoc[faceOrdinal];
          intgExpFace_(scalar_index, 2) = intgLoc_(oppFace_[scalar_index], 2);

          ++scalar_index;
        }
      }
    }
  }

  //left face: x = -1.0 swapped t and u
  faceOrdinal = 3;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      const int oppNode = nodeMap(k,l,1);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index]   = oppNode;
          oppFace_[scalar_index]   = opposing_face_map(l,k,j,i,faceOrdinal);

          intgExpFace_(scalar_index, 0) = faceLoc[faceOrdinal];
          intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_[scalar_index], 1);
          intgExpFace_(scalar_index, 2) = intgLoc_(oppFace_[scalar_index], 2);

          ++scalar_index;
        }
      }
    }
  }

  //bottom face: u = -1.0: swapped s and t
  faceOrdinal = 4;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      const int oppNode = nodeMap(1,k,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opposing_face_map(l,k,j,i,faceOrdinal);

          intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_[scalar_index],0);
          intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_[scalar_index],1);
          intgExpFace_(scalar_index, 2) = faceLoc[faceOrdinal];

          ++scalar_index;
        }
      }
    }
  }

  //top face: u = +1.0: counter-clockwise
  faceOrdinal = 5;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      const int oppNode = nodeMap(nodes1D_-2,l,k);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opposing_face_map(k,l,i,j,faceOrdinal);

          intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_[scalar_index],0);
          intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_[scalar_index],1);
          intgExpFace_(scalar_index, 2) = faceLoc[faceOrdinal];

          ++scalar_index;
        }
      }
    }
  }
}

void
HigherOrderHexSCS::shape_fcn(double* shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int* HigherOrderHexSCS::adjacentNodes()
{
  return &lrscv_(0,0);
}

const int* HigherOrderHexSCS::ipNodeMap(int ordinal) const
{
  return &ipNodeMap_[ordinal*ipsPerFace_];
}

const int *
HigherOrderHexSCS::side_node_ordinals (int ordinal) const
{
  return &sideNodeOrdinals_(ordinal,0);
}

int
HigherOrderHexSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}

int
HigherOrderHexSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}

void
HigherOrderHexSCS::determinant(
  const int  /* nelem */,
  const double *coords,
  double *areav,
  double *error)
{
   constexpr int dim = 3;
   int ipsPerDirection = numIntPoints_ / dim;

   int index = 0;

   //returns the normal vector x_s x x_t for constant u surfaces
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     area_vector<Jacobian::U_DIRECTION>(coords, &shapeDerivs_(index, 0, 0), &areav[index * dim]);
     ++index;
   }

   //returns the normal vector x_u x x_s for constant t surfaces
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_(index, 0, 0), &areav[index * dim]);
     ++index;
   }

   //returns the normal vector x_t x x_u for constant s curves
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_(index, 0, 0), &areav[index * dim]);
     ++index;
   }

   // Multiply with the integration point weighting
   for (int ip = 0; ip < numIntPoints_; ++ip) {
     double weight = ipWeights_[ip];
     areav[ip * dim + 0] *= weight;
     areav[ip * dim + 1] *= weight;
     areav[ip * dim + 2] *= weight;
   }

   *error = 0; // no error checking available
}

template <Jacobian::Direction direction> void
HigherOrderHexSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT areaVector) const
{
  constexpr int s1Component = (direction == Jacobian::T_DIRECTION) ?
      Jacobian::S_DIRECTION : Jacobian::T_DIRECTION;

  constexpr int s2Component = (direction == Jacobian::U_DIRECTION) ?
      Jacobian::S_DIRECTION : Jacobian::U_DIRECTION;

  // return the normal area vector given shape derivatives dnds OR dndt
  double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
  double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDeriv[vector_offset+s1Component];
    const double dn_ds2 = shapeDeriv[vector_offset+s2Component];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
  }

  //cross product
  areaVector[0] = dy_ds1*dz_ds2 - dz_ds1*dy_ds2;
  areaVector[1] = dz_ds1*dx_ds2 - dx_ds1*dz_ds2;
  areaVector[2] = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;
}

void HigherOrderHexSCS::grad_op(
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

    gradient_3d(
      nodesPerElement_,
      coords,
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

void HigherOrderHexSCS::face_grad_op(
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
    gradient_3d(
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

void HigherOrderHexSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(threed_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}

double parametric_distance_hex(const double* x)
{
  std::array<double, 3> y;
  for (int i=0; i<3; ++i) {
    y[i] = std::fabs(x[i]);
  }

  double d = 0;
  for (int i=0; i<3; ++i) {
    if (d < y[i]) {
      d = y[i];
    }
  }
  return d;
}

double HigherOrderHexSCS::isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord)
{
  std::array<double, 3> initialGuess = {{ 0.0, 0.0, 0.0 }};
  int maxIter = 50;
  double tolerance = 1.0e-16;
  double deltaLimit = 1.0e4;

  bool converged = isoparameteric_coordinates_for_point_3d(
      basis_,
      elemNodalCoord,
      pointCoord,
      isoParCoord,
      initialGuess,
      maxIter,
      tolerance,
      deltaLimit
  );
  ThrowAssertMsg(parametric_distance_hex(isoParCoord) < 1.0 + 1.0e-6 || !converged,
      "Inconsistency in parametric distance calculation");

  return (converged) ? parametric_distance_hex(isoParCoord) : std::numeric_limits<double>::max();
}

void HigherOrderHexSCS::interpolatePoint(
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

template <int p> void internal_face_grad_op(
  int face_ordinal,
  const AlignedViewType<DoubleType**[3]>& expReferenceGradWeights,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop )
{
  using traits = AlgTraitsQuadPHexPGL<p>;
  const int offset = traits::numFaceIp_ * face_ordinal;
  auto range = std::make_pair(offset, offset + traits::numFaceIp_);
  auto face_weights = Kokkos::subview(expReferenceGradWeights, range, Kokkos::ALL(), Kokkos::ALL());
  generic_grad_op<AlgTraitsHexGL<p>>(face_weights, coords, gradop);
}

void HigherOrderHexSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop)
{
  switch(nodes1D_ - 1) {
    case 2: return internal_face_grad_op<2>(face_ordinal, expRefGradWeights_, coords, gradop);
    case 3: return internal_face_grad_op<3>(face_ordinal, expRefGradWeights_, coords, gradop);
    case 4: return internal_face_grad_op<4>(face_ordinal, expRefGradWeights_, coords, gradop);
    case USER_POLY_ORDER: return internal_face_grad_op<USER_POLY_ORDER>(face_ordinal, expRefGradWeights_, coords, gradop);
    default: return;
  }
}


}  // namespace nalu
} // namespace sierra
