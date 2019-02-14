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

  //--------------------------------------------------------------------------
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


HigherOrderHexSCV::HigherOrderHexSCV(
  ElementDescription elem,
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
  : MasterElement(),
    elem_(std::move(elem)),
    basis_(std::move(basis)),
    quadrature_(std::move(quadrature))
{
  ndim(elem_.dimension);
  nodesPerElement_ = elem_.nodesPerElement;

  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis.eval_deriv_weights(intgLoc_);
}
//--------------------------------------------------------------------------
void
HigherOrderHexSCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (elem_.nodes1D * elem_.nodes1D  * elem_.nodes1D)
                * ( quadrature_.num_quad() * quadrature_.num_quad() * quadrature_.num_quad());

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*nDim_);
  intgLocShift_.resize(numIntPoints_*nDim_);
  ipWeights_.resize(numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int vector_index = 0; int scalar_index = 0;
  for (int n = 0; n < elem_.nodes1D; ++n) {
    for (int m = 0; m < elem_.nodes1D; ++m) {
      for (int l = 0; l < elem_.nodes1D; ++l) {
        for (int k = 0; k < quadrature_.num_quad(); ++k) {
          for (int j = 0; j < quadrature_.num_quad(); ++j) {
            for (int i = 0; i < quadrature_.num_quad(); ++i) {
              intgLoc_[vector_index + 0] = quadrature_.integration_point_location(l,i);
              intgLoc_[vector_index + 1] = quadrature_.integration_point_location(m,j);
              intgLoc_[vector_index + 2] = quadrature_.integration_point_location(n,k);
              ipWeights_[scalar_index] = quadrature_.integration_point_weight(l,m,n,i,j,k);
              ipNodeMap_[scalar_index] = elem_.node_map(l, m, n);

              ++scalar_index;
              vector_index += nDim_;
            }
          }
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderHexSCV::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderHexSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}
//--------------------------------------------------------------------------
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
    const double det_j = jacobian_determinant( coords,  &shapeDerivs_[grad_offset]);
    volume[ip] = ipWeights_[ip] * det_j;

    if (det_j < tiny_positive_value()) {
      *error = 1.0;
    }
  }
}
//--------------------------------------------------------------------------
double
HigherOrderHexSCV::jacobian_determinant(
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
//--------------------------------------------------------------------------
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
      deriv[grad_offset + j] = shapeDerivs_[grad_offset +j];
    }

    gradient_3d(
      nodesPerElement_,
      coords,
      &shapeDerivs_[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

//--------------------------------------------------------------------------
int ip_per_face(const TensorProductQuadratureRule& quad, const LagrangeBasis& basis) {
  return quad.num_quad() * quad.num_quad() * (basis.polyOrder_+1)*(basis.polyOrder_+1);
}
//--------------------------------------------------------------------------
HigherOrderHexSCS::HigherOrderHexSCS(
  ElementDescription elem,
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  elem_(std::move(elem)),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
  expRefGradWeights_("reference_gradient_weights", 6*ip_per_face(quadrature,basis), basis.num_nodes())
{
  ndim(elem_.dimension);
  nodesPerElement_ = elem_.nodesPerElement;

  // set up integration rule and relevant maps on scs
  set_interior_info();

  // set up integration rule and relevant maps on faces
  set_boundary_info();

  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = basis_.eval_deriv_weights(intgExpFace_);

  {
    const int numFaceIps = 6*ip_per_face(quadrature,basis);
    int deriv_index = 0;
    for (int ip = 0; ip < numFaceIps; ++ip) {
      for (int n = 0; n < nodesPerElement_; ++n) {
        for (int d = 0; d < 3; ++d) {
          expRefGradWeights_(ip,n,d) = expFaceShapeDerivs_[deriv_index];
          ++deriv_index;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderHexSCS::set_interior_info()
{
  const int surfacesPerDirection = elem_.nodes1D - 1;
  const int ipsPerSurface = (quadrature_.num_quad() * quadrature_.num_quad()) * (elem_.nodes1D * elem_.nodes1D);
  const int numSurfaces = surfacesPerDirection * nDim_;

  numIntPoints_ = numSurfaces*ipsPerSurface;
  const int numVectorPoints = numIntPoints_*nDim_;

  // define L/R mappings
  lrscv_.resize(2*numIntPoints_);

  // standard integration location
  intgLoc_.resize(numVectorPoints);

  // shifted
  intgLocShift_.resize(numVectorPoints);

  // Save quadrature weight and directionality information
  ipInfo_.resize(numIntPoints_);

  // specify integration point locations in a dimension-by-dimension manner
  // u direction: bottom-top (0-1)
  int vector_index = 0; int lrscv_index = 0; int scalar_index = 0;
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < elem_.nodes1D; ++l) {
      for (int k = 0; k < elem_.nodes1D; ++k) {

        int leftNode; int rightNode; int orientation;
        if (m % 2 == 0) {
          leftNode = elem_.node_map(k,l,m);
          rightNode = elem_.node_map(k,l,m+1);
          orientation = -1;
        }
        else {
          leftNode = elem_.node_map(k,l,m+1);
          rightNode = elem_.node_map(k,l,m);
          orientation = +1;
        }

        for (int j = 0; j < quadrature_.num_quad(); ++j) {
          for (int i = 0; i < quadrature_.num_quad(); ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = quadrature_.integration_point_location(k,i);
            intgLoc_[vector_index + 1] = quadrature_.integration_point_location(l,j);
            intgLoc_[vector_index + 2] = quadrature_.scs_loc(m);

            ipInfo_[scalar_index].weight = orientation * quadrature_.integration_point_weight(k, l, i, j);

            ipInfo_[scalar_index].direction = Jacobian::U_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }

  // t direction: front-back (2-3)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < elem_.nodes1D; ++l) {
      for (int k = 0; k < elem_.nodes1D; ++k) {

        int leftNode; int rightNode; int orientation;
        if (m % 2 == 0) {
          leftNode  = elem_.node_map(k,m+0,l);
          rightNode = elem_.node_map(k,m+1,l);
          orientation = -1;
        }
        else {
          leftNode  = elem_.node_map(k,m+1,l);
          rightNode = elem_.node_map(k,m+0,l);
          orientation = +1;
        }

        for (int j = 0; j < quadrature_.num_quad(); ++j) {
          for (int i = 0; i < quadrature_.num_quad(); ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = quadrature_.integration_point_location(k,i);
            intgLoc_[vector_index + 1] = quadrature_.scs_loc(m);
            intgLoc_[vector_index + 2] = quadrature_.integration_point_location(l,j);

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = orientation * quadrature_.integration_point_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::T_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }

  //s direction: left-right (4-5)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < elem_.nodes1D; ++l) {
      for (int k = 0; k < elem_.nodes1D; ++k) {

        int leftNode; int rightNode; int orientation;
        if (m % 2 == 0) {
          leftNode  = elem_.node_map(m+0,k,l);
          rightNode = elem_.node_map(m+1,k,l);
          orientation = +1;
        }
        else {
          leftNode  = elem_.node_map(m+1,k,l);
          rightNode = elem_.node_map(m+0,k,l);
          orientation = -1;
        }

        for (int j = 0; j < quadrature_.num_quad(); ++j) {
          for (int i = 0; i < quadrature_.num_quad(); ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = quadrature_.scs_loc(m);
            intgLoc_[vector_index + 1] = quadrature_.integration_point_location(k,i);
            intgLoc_[vector_index + 2] = quadrature_.integration_point_location(l,j);

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = orientation * quadrature_.integration_point_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::S_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderHexSCS::set_boundary_info()
{
  const int numFaces = elem_.numFaces;
  int numQuad = static_cast<int>(quadrature_.num_quad());
  int nodes1D = static_cast<int>(elem_.nodes1D);
  const int nodesPerFace = nodes1D * nodes1D;
  ipsPerFace_ = nodesPerFace * (numQuad * numQuad);
  int surfacesPerDirection = nodes1D - 1;
  const int numFaceIps = numFaces * ipsPerFace_;

  oppFace_.resize(numFaceIps);
  ipNodeMap_.resize(numFaceIps);
  oppNode_.resize(numFaceIps);
  intgExpFace_.resize(numFaceIps*nDim_);

  // tensor-product style access to the map
  auto face_node_number = [&] (int i, int j, int faceOrdinal)
  {
    return elem_.faceNodeMap[faceOrdinal][i + nodes1D * j];
  };

  // map face ip ordinal to nearest sub-control surface ip ordinal
  // sub-control surface renumbering
  const std::vector<int> faceToSurface = {
      surfacesPerDirection,     // nearest scs face to t=-1.0
      3*surfacesPerDirection-1, // nearest scs face to s=+1.0, the last face
      2*surfacesPerDirection-1, // nearest scs face to t=+1.0
      2*surfacesPerDirection,   // nearest scs face to s=-1.0
      0,                        // nearest scs face to u=-1.0, the first face
      surfacesPerDirection-1    // nearest scs face to u=+1.0, the first face
  };

  auto opp_face_map = [&] ( int k, int l, int i, int j, int face_index)
  {
    int face_offset = faceToSurface[face_index] * ipsPerFace_;

    int node_index = k + nodes1D * l;
    int node_offset = node_index * (numQuad * numQuad);

    int ip_index = face_offset+node_offset+i+numQuad*j;

    return ip_index;
  };

  // location of the faces in the correct order
  const std::vector<double> faceLoc = {-1.0, +1.0, +1.0, -1.0, -1.0, +1.0};

  // Set points face-by-face
  int vector_index = 0; int scalar_index = 0; int faceOrdinal = 0;

  // front face: t = -1.0: counter-clockwise
  faceOrdinal = 0;
  for (int l = 0; l < nodes1D; ++l) {
    for (int k = 0; k < nodes1D; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = elem_.node_map(k,1,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad; ++j) {
        for (int i = 0; i < numQuad; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  // right face: s = +1.0: counter-clockwise
  faceOrdinal = 1;
  for (int l = 0; l < nodes1D; ++l) {
    for (int k = 0; k < nodes1D; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = elem_.node_map(nodes1D-2,k,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad; ++j) {
        for (int i = 0; i < numQuad; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  // back face: t = +1.0: s-direction reversed
  faceOrdinal = 2;
  for (int l = 0; l < nodes1D; ++l) {
    for (int k = nodes1D-1; k >= 0; --k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = elem_.node_map(k,nodes1D-2,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad; ++j) {
        for (int i = numQuad-1; i >= 0; --i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //left face: x = -1.0 swapped t and u
  faceOrdinal = 3;
  for (int l = 0; l < nodes1D; ++l) {
    for (int k = 0; k < nodes1D; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      int oppNode = elem_.node_map(1,l,k);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad; ++j) {
        for (int i = 0; i < numQuad; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index]   = oppNode;
          oppFace_[scalar_index]   = opp_face_map(l,k,j,i,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //bottom face: u = -1.0: swapped s and t
  faceOrdinal = 4;
  for (int l = 0; l < nodes1D; ++l) {
    for (int k = 0; k < nodes1D; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      int oppNode = elem_.node_map(l,k,1);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad; ++j) {
        for (int i = 0; i < numQuad; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(l,k,j,i,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = faceLoc[faceOrdinal];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //top face: u = +1.0: counter-clockwise
  faceOrdinal = 5;
  for (int l = 0; l < nodes1D; ++l) {
    for (int k = 0; k < nodes1D; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = elem_.node_map(k,l,nodes1D-2);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad; ++j) {
        for (int i = 0; i < numQuad; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = faceLoc[faceOrdinal];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderHexSCS::shape_fcn(double* shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderHexSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}
//--------------------------------------------------------------------------
const int *
HigherOrderHexSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*ipsPerFace_];
}
//--------------------------------------------------------------------------
const int *
HigherOrderHexSCS::side_node_ordinals(int ordinal)
{
  return elem_.side_node_ordinals(ordinal);
}
//--------------------------------------------------------------------------
int
HigherOrderHexSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}
//--------------------------------------------------------------------------
int
HigherOrderHexSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}
//--------------------------------------------------------------------------
void
HigherOrderHexSCS::determinant(
  const int  /* nelem */,
  const double *coords,
  double *areav,
  double *error)
{
   constexpr int dim = 3;
   int ipsPerDirection = numIntPoints_ / dim;
   int deriv_increment = dim * nodesPerElement_;

   int index = 0;

   //returns the normal vector x_s x x_t for constant u surfaces
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     ThrowAssert(ipInfo_[index].direction == Jacobian::U_DIRECTION);
     area_vector<Jacobian::U_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
     ++index;
   }

   //returns the normal vector x_u x x_s for constant t surfaces
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     ThrowAssert(ipInfo_[index].direction == Jacobian::T_DIRECTION);
     area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
     ++index;
   }

   //returns the normal vector x_t x x_u for constant s curves
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     ThrowAssert(ipInfo_[index].direction == Jacobian::S_DIRECTION);
     area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
     ++index;
   }

   // Multiply with the integration point weighting
   for (int ip = 0; ip < numIntPoints_; ++ip) {
     double weight = ipInfo_[ip].weight;
     areav[ip * dim + 0] *= weight;
     areav[ip * dim + 1] *= weight;
     areav[ip * dim + 2] *= weight;
   }

   *error = 0; // no error checking available
}
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
      deriv[grad_offset + j] = shapeDerivs_[grad_offset +j];
    }

    gradient_3d(
      nodesPerElement_,
      coords,
      &shapeDerivs_[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
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
  const double* const faceShapeDerivs = &expFaceShapeDerivs_[face_offset];

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

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop )
{
  using traits = AlgTraitsQuadPHexPGL<p>;
  const int offset = traits::numFaceIp_ * face_ordinal;
  auto range = std::make_pair(offset, offset + traits::numFaceIp_);
  auto face_weights = Kokkos::subview(expReferenceGradWeights, range, Kokkos::ALL(), Kokkos::ALL());
  generic_grad_op<AlgTraitsHexGL<p>>(face_weights, coords, gradop);
}

void HigherOrderHexSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  switch(elem_.polyOrder) {
    case 2: return internal_face_grad_op<2>(face_ordinal, expRefGradWeights_, coords, gradop);
    case 3: return internal_face_grad_op<3>(face_ordinal, expRefGradWeights_, coords, gradop);
    case 4: return internal_face_grad_op<4>(face_ordinal, expRefGradWeights_, coords, gradop);
    case USER_POLY_ORDER: return internal_face_grad_op<USER_POLY_ORDER>(face_ordinal, expRefGradWeights_, coords, gradop);
    default: {
      throw std::runtime_error("Invalid poly order " + std::to_string(elem_.polyOrder) + " for kernels");
      return;
    }
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HigherOrderQuad3DSCS::HigherOrderQuad3DSCS(
  ElementDescription elem,
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  elem_(std::move(elem)),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature))
{
  surfaceDimension_ = 2;
  ndim(3);
  nodesPerElement_ = elem_.nodes1D * elem_.nodes1D;

  // set up integration rule and relevant maps on scs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis.eval_deriv_weights(intgLoc_);
}
//--------------------------------------------------------------------------
void
HigherOrderQuad3DSCS::set_interior_info()
{
  nodesPerElement_ = elem_.nodes1D * elem_.nodes1D;

  //1D integration rule per sub-control volume
  numIntPoints_ = (elem_.nodes1D * elem_.nodes1D) * ( quadrature_.num_quad() * quadrature_.num_quad() );

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*surfaceDimension_);
  intgLocShift_.resize(numIntPoints_*surfaceDimension_);
  ipWeights_.resize(numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int vector_index_2D = 0; int scalar_index = 0;
  for (int l = 0; l < elem_.nodes1D; ++l) {
    for (int k = 0; k < elem_.nodes1D; ++k) {
      for (int j = 0; j < quadrature_.num_quad(); ++j) {
        for (int i = 0; i < quadrature_.num_quad(); ++i) {
          //integration point location
          intgLoc_[vector_index_2D]     = quadrature_.integration_point_location(k,i);
          intgLoc_[vector_index_2D + 1] = quadrature_.integration_point_location(l,j);

          //weight
          ipWeights_[scalar_index] = quadrature_.integration_point_weight(k,l,i,j);

          //sub-control volume association
          ipNodeMap_[scalar_index] = elem_.node_map_bc(k,l);

          // increment indices
          ++scalar_index;
          vector_index_2D += surfaceDimension_;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad3DSCS::shape_fcn(double* shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad3DSCS::ipNodeMap(
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal);
  return &ipNodeMap_[0];
}
//--------------------------------------------------------------------------
void
HigherOrderQuad3DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double * /*error*/)
{
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  std::array<double, 3> areaVector;
  int grad_offset = 0;
  int grad_inc = surfaceDimension_ * nodesPerElement_;

  int vector_offset = 0;
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    //compute area vector for this ip
    area_vector( &coords[0], &shapeDerivs_[grad_offset], areaVector );

    // apply quadrature weight and orientation (combined as weight)
    for (int j = 0; j < nDim_; ++j) {
      areav[vector_offset+j]  = ipWeights_[ip] * areaVector[j];
    }
    vector_offset += nDim_;
    grad_offset += grad_inc;
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad3DSCS::area_vector(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  std::array<double,3>& areaVector) const
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
  areaVector[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
  areaVector[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
  areaVector[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}
//--------------------------------------------------------------------------
HigherOrderQuad2DSCV::HigherOrderQuad2DSCV(
  ElementDescription elem,
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  elem_(std::move(elem)),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature))
{
  ndim(elem_.dimension);
  nodesPerElement_ = elem_.nodesPerElement;

  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis.eval_deriv_weights(intgLoc_);

}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (elem_.nodes1D * elem_.nodes1D) * ( quadrature_.num_quad() * quadrature_.num_quad() );

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*nDim_);
  intgLocShift_.resize(numIntPoints_*nDim_);
  ipWeights_.resize(numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int vector_index = 0; int scalar_index = 0;
  for (int l = 0; l < elem_.nodes1D; ++l) {
    for (int k = 0; k < elem_.nodes1D; ++k) {
      for (int j = 0; j < quadrature_.num_quad(); ++j) {
        for (int i = 0; i < quadrature_.num_quad(); ++i) {
          intgLoc_[vector_index]     = quadrature_.integration_point_location(k,i);
          intgLoc_[vector_index + 1] = quadrature_.integration_point_location(l,j);
          ipWeights_[scalar_index] = quadrature_.integration_point_weight(k,l,i,j);
          ipNodeMap_[scalar_index] = elem_.node_map(k,l);

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCV::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCV::ipNodeMap(int /*ordinal*/)
{
  return &ipNodeMap_[0];
}
//--------------------------------------------------------------------------
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
    const double det_j = jacobian_determinant(coords, &shapeDerivs_[grad_offset] );
    volume[ip] = ipWeights_[ip] * det_j;

    //flag error
    if (det_j < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
      deriv[grad_offset + j] = shapeDerivs_[grad_offset +j];
    }

    gradient_2d(
      nodesPerElement_,
      &coords[0],
      &shapeDerivs_[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }
    grad_offset += grad_inc;
  }
}
//--------------------------------------------------------------------------
HigherOrderQuad2DSCS::HigherOrderQuad2DSCS(
  ElementDescription elem,
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  elem_(std::move(elem)),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature))
{
  ndim(elem_.dimension);
  nodesPerElement_ = elem_.nodesPerElement;

  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = basis.eval_deriv_weights(intgExpFace_);
}

double parametric_distance_quad(const double* x)
{
  double absXi  = std::abs(x[0]);
  double absEta = std::abs(x[1]);
  return (absXi > absEta) ? absXi : absEta;
}

//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::set_interior_info()
{
  const int linesPerDirection = elem_.nodes1D - 1;
  const int ipsPerLine = quadrature_.num_quad() * elem_.nodes1D;
  const int numLines = linesPerDirection * nDim_;

  numIntPoints_ = numLines * ipsPerLine;

  // define L/R mappings
  lrscv_.resize(2*numIntPoints_);

  // standard integration location
  intgLoc_.resize(numIntPoints_*nDim_);

  // shifted
  intgLocShift_.resize(numIntPoints_*nDim_);

  ipInfo_.resize(numIntPoints_);

  // specify integration point locations in a dimension-by-dimension manner

  //u-direction
  int vector_index = 0;
  int lrscv_index = 0;
  int scalar_index = 0;
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < elem_.nodes1D; ++l) {

      int leftNode;
      int rightNode;
      int orientation;
      if (m % 2 == 0) {
        leftNode  = elem_.node_map(l,m + 0);
        rightNode = elem_.node_map(l,m + 1);
        orientation = -1;
      }
      else {
        leftNode  = elem_.node_map(l,m + 1);
        rightNode = elem_.node_map(l,m + 0);
        orientation = +1;
      }

      for (int j = 0; j < quadrature_.num_quad(); ++j) {

        lrscv_[lrscv_index] = leftNode;
        lrscv_[lrscv_index + 1] = rightNode;

        intgLoc_[vector_index] = quadrature_.integration_point_location(l,j);
        intgLoc_[vector_index + 1] = quadrature_.scs_loc(m);

        //compute the quadrature weight
        ipInfo_[scalar_index].weight = orientation*quadrature_.integration_point_weight(l,j);

        //direction
        ipInfo_[scalar_index].direction = Jacobian::T_DIRECTION;

        ++scalar_index;
        lrscv_index += 2;
        vector_index += nDim_;
      }
    }
  }

  //t-direction
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < elem_.nodes1D; ++l) {

      int leftNode;
      int rightNode;
      int orientation;
      if (m % 2 == 0) {
        leftNode  = elem_.node_map(m,l);
        rightNode = elem_.node_map(m+1,l);
        orientation = +1;
      }
      else {
        leftNode  = elem_.node_map(m+1,l);
        rightNode = elem_.node_map(m,l);
        orientation = -1;
      }

      for (int j = 0; j < quadrature_.num_quad(); ++j) {

        lrscv_[lrscv_index]   = leftNode;
        lrscv_[lrscv_index+1] = rightNode;

        intgLoc_[vector_index] = quadrature_.scs_loc(m);
        intgLoc_[vector_index+1] = quadrature_.integration_point_location(l,j);

        //compute the quadrature weight
        ipInfo_[scalar_index].weight = orientation*quadrature_.integration_point_weight(l,j);

        //direction
        ipInfo_[scalar_index].direction = Jacobian::S_DIRECTION;

        ++scalar_index;
        lrscv_index += 2;
        vector_index += nDim_;
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::set_boundary_info()
{
  const int numFaces = 2*nDim_;
  const int nodesPerFace = elem_.nodes1D;
  const int linesPerDirection = elem_.nodes1D-1;
  ipsPerFace_ = nodesPerFace*quadrature_.num_quad();

  const int numFaceIps = numFaces*ipsPerFace_;

  oppFace_.resize(numFaceIps);
  ipNodeMap_.resize(numFaceIps);
  oppNode_.resize(numFaceIps);
  intgExpFace_.resize(numFaceIps*nDim_);

  auto faceMap = elem_.faceNodeMap;
  auto nodeMap = elem_.nodeMap;
  auto nodeMap1D = elem_.nodeMapBC;

  auto face_node_number = [&] (int number,int faceOrdinal)
  {
    return elem_.faceNodeMap[faceOrdinal][number];
  };

  const std::array<int, 4> faceToLine = {{
      0,
      2*linesPerDirection-1,
      linesPerDirection-1,
      linesPerDirection
  }};

  const std::array<double, 4> faceLoc = {{-1.0, +1.0, +1.0, -1.0}};

  int scalar_index = 0; int vector_index = 0;
  int faceOrdinal = 0; //bottom face
  int oppFaceIndex = 0;
  for (int k = 0; k < elem_.nodes1D; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = elem_.node_map(k, 1);

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = intgLoc_[oppFace_[scalar_index]*nDim_+0];
      intgExpFace_[vector_index+1] = faceLoc[faceOrdinal];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 1; //right face
  oppFaceIndex = 0;
  for (int k = 0; k < elem_.nodes1D; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = elem_.node_map(elem_.nodes1D-2,k);

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = faceLoc[faceOrdinal];
      intgExpFace_[vector_index+1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }


  faceOrdinal = 2; //top face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  int elemNodeM1 = static_cast<int>(elem_.nodes1D-1);
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(elem_.nodes1D-k-1,faceOrdinal);
    int oppNode = elem_.node_map(k, elem_.nodes1D-2);
    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index] = intgLoc_[oppFace_[scalar_index]*nDim_+0];
      intgExpFace_[vector_index+1] = faceLoc[faceOrdinal];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 3; //left face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(elem_.nodes1D-k-1,faceOrdinal);
    int oppNode = elem_.node_map(1,k);
    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = faceLoc[faceOrdinal];
      intgExpFace_[vector_index+1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCS::ipNodeMap(int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*ipsPerFace_];
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCS::side_node_ordinals(int ordinal)
{
  return elem_.side_node_ordinals(ordinal);
}
//--------------------------------------------------------------------------
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
  int deriv_increment = dim * nodesPerElement_;

  int index = 0;

   //returns the normal vector x_u x x_s for constant t surfaces
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::T_DIRECTION);
    area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
    ++index;
  }

  //returns the normal vector x_t x x_u for constant s curves
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::S_DIRECTION);
    area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
    ++index;
  }

  // Multiply with the integration point weighting
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    double weight = ipInfo_[ip].weight;
    areav[ip * dim + 0] *= weight;
    areav[ip * dim + 1] *= weight;
  }
}
//--------------------------------------------------------------------------
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
      deriv[grad_offset + j] = shapeDerivs_[grad_offset +j];
    }

    gradient_2d(
      nodesPerElement_,
      &coords[0],
      &shapeDerivs_[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}
//--------------------------------------------------------------------------
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
  const double* const faceShapeDerivs = &expFaceShapeDerivs_[face_offset];

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

//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}
//--------------------------------------------------------------------------
int
HigherOrderQuad2DSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}
//--------------------------------------------------------------------------
int
HigherOrderQuad2DSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
HigherOrderEdge2DSCS::HigherOrderEdge2DSCS(
  ElementDescription elem,
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  elem_(elem),
  basis_(basis),
  quadrature_(quadrature)
{
  ndim(2);
  nodesPerElement_ = elem_.nodes1D;
  numIntPoints_ = quadrature_.num_quad() * elem_.nodes1D;

  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_);

  ipWeights_.resize(numIntPoints_);

  int scalar_index = 0;
  for (int k = 0; k < elem_.nodes1D; ++k) {
    for (int i = 0; i < quadrature_.num_quad(); ++i) {
      intgLoc_[scalar_index]  = quadrature_.integration_point_location(k,i);
      ipWeights_[scalar_index] = quadrature_.integration_point_weight(k,i);
      ipNodeMap_[scalar_index] = elem_.node_map_bc(k);
      ++scalar_index;
    }
  }

  shapeFunctionVals_ = basis.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis.eval_deriv_weights(intgLoc_);
}
//--------------------------------------------------------------------------
const int *
HigherOrderEdge2DSCS::ipNodeMap(int /*ordinal*/)
{
  return &ipNodeMap_[0];
}
//--------------------------------------------------------------------------
void
HigherOrderEdge2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  std::array<double,2> areaVector;
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  int grad_offset = 0;
  const int grad_inc = nodesPerElement_;

  int vec_offset = 0;
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    // calculate the area vector
    area_vector( &coords[0],
      &shapeDerivs_[grad_offset],
      areaVector );

    // weight the area vector with the Gauss-quadrature weight for this IP
    areav[vec_offset + 0] = ipWeights_[ip] * areaVector[0];
    areav[vec_offset + 1] = ipWeights_[ip] * areaVector[1];

    grad_offset += grad_inc;
    vec_offset += nDim_;
  }
}
//--------------------------------------------------------------------------
void
HigherOrderEdge2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
   for (int j = 0; j < numShape; ++j) {
     shpfc[j] = shapeFunctionVals_[j];
   }
}
//--------------------------------------------------------------------------
void
HigherOrderEdge2DSCS::area_vector(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  std::array<double,2>& areaVector) const
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
  areaVector[0] =  dydr;
  areaVector[1] = -dxdr;
}

}  // namespace nalu
} // namespace sierra
