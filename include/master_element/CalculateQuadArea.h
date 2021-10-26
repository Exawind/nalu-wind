// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CalculateQuadArea_h
#define CalculateQuadArea_h

#include <AlgTraits.h>
#include <NaluEnv.h>

namespace sierra{
namespace nalu{

//    This subroutine computes the area of a quadrilateral surface by
//    decomposing it into four triangles and summing the contribution
//    of each triangle.
//
//    3D coordinates of each vertex of the face
//    =========================================
//    double areacoords(n_v3d,4)
//
//    area output
//    ===========
//    double area(3)

KOKKOS_FUNCTION
void quadAreaByTriangleFacets(const double *areacoords, double *area);

} // namespace nalu
} // namespace Sierra

#endif
