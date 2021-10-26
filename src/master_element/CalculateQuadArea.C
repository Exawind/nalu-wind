// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/CalculateQuadArea.h>

namespace sierra{
namespace nalu{

//     This subroutine computes the area of a quadrilateral surface by
//     decomposing it into four triangles and summing the contribution
//     of each triangle.
//
//     3D coordinates of each vertex of the face
//     =========================================
//     double areacoords(n_v3d,4)
//
//     area output
//     ===========
//     double area(3)
void quadAreaByTriangleFacets(
  const double * areacoords, 
  double * area )
{       
  auto idx = [&] (const int k, const int j)
  {
    return 4*k + j;
  };
 
  // this table defines the triangles composing the face with
  // the convention that all cross products will point the
  // same direction by convention
  const int triangularFacetTable[4][3] = 
    {{4, 0, 1}, {4, 1, 2}, {4, 2, 3}, {4, 3, 0}};

  const double one4th = 1.0/4.0;
  const double half = 1.0/2.0;

  double xmid[3];
  for (int k=0; k<3; ++k)
    xmid[k] = one4th*
         ( areacoords[idx(k,0)] + areacoords[idx(k,1)]
         + areacoords[idx(k,2)] + areacoords[idx(k,3)]);
  double r2[3];
  for (int k=0; k<3; ++k)
  {
    area[k] = 0;
    r2[k] = areacoords[idx(k,0)] - xmid[k];
  }
  // loop over triangles
  const int ntriangles = 4;
  for (int itriangle = 0; itriangle < ntriangles; ++itriangle)
  {
    double r1[3];
    const int iq = triangularFacetTable[itriangle][2];
    // construct vectors with common beginning point
    for (int k=0; k<3; ++k) 
    {
      r1[k] = r2[k];
      r2[k] = areacoords[idx(k,iq)]-xmid[k];
    }
    // cross product is twice the area vector
    // triangularFaceTable should be constructed such
    // that these vectors have the same convention (right hand rule)
    area[0] += r1[1]*r2[2] - r2[1]*r1[2];
    area[1] += r1[2]*r2[0] - r2[2]*r1[0];
    area[2] += r1[0]*r2[1] - r2[0]*r1[1];
  }
  // apply the 1/2 that was pulled out
  for (int k=0; k<3; ++k) 
    area[k] *= half;
}
}
}
