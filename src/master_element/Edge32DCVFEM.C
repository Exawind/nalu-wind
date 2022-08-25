// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/Edge32DCVFEM.h>
#include <master_element/MasterElementFunctions.h>
#include <AlgTraits.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Edge32DSCS::Edge32DSCS() : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

#ifndef KOKKOS_ENABLE_CUDA
  const int stk1DNodeMap[3] = {0, 2, 1};

  int scalar_index = 0;
  for (int k = 0; k < nodesPerElement_; ++k) {
    for (int i = 0; i < numQuad_; ++i) {
      ipNodeMap_[scalar_index] = stk1DNodeMap[k];
      intgLoc_[scalar_index] = gauss_point_location(k, i);
      ipWeight_[scalar_index] = tensor_product_weight(k, i);
      ++scalar_index;
    }
  }
#endif
}

//--------------------------------------------------------------------------
//-------- gauss_point_location --------------------------------------------
//--------------------------------------------------------------------------
double
Edge32DSCS::gauss_point_location(
  const int nodeOrdinal, const int gaussPointOrdinal) const
{
  return isoparametric_mapping(
    scsEndLoc_[nodeOrdinal + 1], scsEndLoc_[nodeOrdinal],
    gaussAbscissae_[gaussPointOrdinal]);
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------
double
Edge32DSCS::tensor_product_weight(const int s1Node, const int s1Ip) const
{
  // line integration
  const double isoparametricLength =
    scsEndLoc_[s1Node + 1] - scsEndLoc_[s1Node];
  const double weight = isoparametricLength * gaussWeight_[s1Ip];
  return weight;
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int*
Edge32DSCS::ipNodeMap(int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
Edge32DSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE**, SHMEM>& area) const
{

  NALU_ALIGNED DBLTYPE areaVector[2];
  const DBLTYPE x0 = coords(0, 0);
  const DBLTYPE y0 = coords(0, 1);
  const DBLTYPE x1 = coords(1, 0);
  const DBLTYPE y1 = coords(1, 1);
  const DBLTYPE x2 = coords(2, 0);
  const DBLTYPE y2 = coords(2, 1);

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    const DBLTYPE s = intgLoc_[ip];
    const DBLTYPE dxds = 0.5 * (x1 - x0) + (x1 - 2.0 * x2 + x0) * s;
    const DBLTYPE dyds = 0.5 * (y1 - y0) + (y1 - 2.0 * y2 + y0) * s;

    areaVector[0] = dyds;
    areaVector[1] = -dxds;

    // weight the area vector with the Gauss-quadrature weight for this IP
    area(ip, 0) = ipWeight_[ip] * areaVector[0];
    area(ip, 1) = ipWeight_[ip] * areaVector[1];
  }
}

KOKKOS_FUNCTION void
Edge32DSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType**, DeviceShmem>& area)
{
  determinant_scs(coords, area);
}
void
Edge32DSCS::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double**>& area)
{
  determinant_scs(coords, area);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  for (int i = 0; i < numIntPoints_; ++i) {
    const DoubleType s = intgLoc_[i];
    shpfc(i, 0) = -s * (1.0 - s) * 0.5;
    shpfc(i, 1) = s * (1.0 + s) * 0.5;
    shpfc(i, 2) = (1.0 - s) * (1.0 + s);
  }
}

void
Edge32DSCS::shape_fcn(double* shpfc)
{
  for (int i = 0; i < numIntPoints_; ++i) {
    int j = 3 * i;
    const double s = intgLoc_[i];
    shpfc[j] = -s * (1.0 - s) * 0.5;
    shpfc[j + 1] = s * (1.0 + s) * 0.5;
    shpfc[j + 2] = (1.0 - s) * (1.0 + s);
  }
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  for (int i = 0; i < numIntPoints_; ++i) {
    const DoubleType s = intgLocShift_[i];
    shpfc(i, 0) = -s * (1.0 - s) * 0.5;
    shpfc(i, 1) = s * (1.0 + s) * 0.5;
    shpfc(i, 2) = (1.0 - s) * (1.0 + s);
  }
}

void
Edge32DSCS::shifted_shape_fcn(double* shpfc)
{
  for (int i = 0; i < numIntPoints_; ++i) {
    int j = 3 * i;
    const double s = intgLocShift_[i];
    shpfc[j] = -s * (1.0 - s) * 0.5;
    shpfc[j + 1] = s * (1.0 + s) * 0.5;
    shpfc[j + 2] = (1.0 - s) * (1.0 + s);
  }
}

//--------------------------------------------------------------------------
//-------- interpolate_point -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::interpolatePoint(
  const int& nComp,
  const double* isoParCoord,
  const double* field,
  double* result)
{
  constexpr int nNodes = 3;

  double s = isoParCoord[0];
  std::array<double, nNodes> shapefct = {
    {-0.5 * s * (1 - s), +0.5 * s * (1 + s), (1 - s) * (1 + s)}};

  for (int i = 0; i < nComp; ++i) {
    result[i] = shapefct[0] * field[3 * i + 0] +
                shapefct[1] * field[3 * i + 1] + shapefct[2] * field[3 * i + 2];
  }
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::area_vector(
  const double* POINTER_RESTRICT coords,
  const double s,
  double* POINTER_RESTRICT areaVector) const
{
  // returns the normal area vector (dyds,-dxds) evaluated at s

  // create a parameterization of the curve
  // r(s) = (x(s),y(s)) s.t. r(-1) = (x0,y0); r(0) = (x2,y2); r(1) = (x1,y1);
  // x(s) = x2 + 0.5 (x1-x0) s + 0.5 (x1 - 2 x2 + x0) s^2,
  // y(s) = y2 + 0.5 (y1-y0) s + 0.5 (y1 - 2 y2 + y0) s^2
  // could equivalently use the shape function derivatives . . .

  // coordinate names
  const double x0 = coords[0];
  const double y0 = coords[1];
  const double x1 = coords[2];
  const double y1 = coords[3];
  const double x2 = coords[4];
  const double y2 = coords[5];

  const double dxds = 0.5 * (x1 - x0) + (x1 - 2.0 * x2 + x0) * s;
  const double dyds = 0.5 * (y1 - y0) + (y1 - 2.0 * y2 + y0) * s;

  areaVector[0] = dyds;
  areaVector[1] = -dxds;
}

} // namespace nalu
} // namespace sierra
