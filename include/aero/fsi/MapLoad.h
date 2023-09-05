// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MAP_LOAD_H
#define MAP_LOAD_H

#include <KokkosInterface.h>
#include <stk_mesh/base/BulkData.hpp>
#include <FieldTypeDef.h>
#include <vector>

namespace fsi {

//! Linearly interpolate between 3-dimensional vectors 'a' and 'b' with
//! interpolating factor 'interpFac'
void KOKKOS_INLINE_FUNCTION
linInterpVec(double* a, double* b, double interpFac, double* aInterpb)
{
  for (size_t i = 0; i < 3; i++)
    aInterpb[i] = a[i] + interpFac * (b[i] - a[i]);
}

void mapTowerLoad(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& twrBndyParts,
  const sierra::nalu::VectorFieldType& modelCoords,
  const sierra::nalu::VectorFieldType& meshDisp,
  sierra::nalu::GenericIntFieldType& loadMap,
  sierra::nalu::GenericFieldType& loadMapInterp,
  sierra::nalu::GenericFieldType& tforceSCS,
  std::vector<double>& twrRefPos,
  std::vector<double>& twrDef,
  std::vector<double>& twrLoad);

void mapBladeLoad(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& twrBndyParts,
  const sierra::nalu::VectorFieldType& modelCoords,
  const sierra::nalu::VectorFieldType& meshDisp,
  sierra::nalu::GenericIntFieldType& loadMap,
  sierra::nalu::GenericFieldType& loadMapInterp,
  sierra::nalu::GenericFieldType& tforceSCS,
  int nPtsBlade,
  int iStart,
  std::vector<double>& bldRloc,
  std::vector<double>& bldRefPos,
  std::vector<double>& bldDef,
  std::vector<double>& bldLoad);

} // namespace fsi

#endif
