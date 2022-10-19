// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MESHVELOCITYALG_H
#define MESHVELOCITYALG_H

#include "KokkosInterface.h"
#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

template <typename AlgTraits>
class MeshVelocityAlg : public Algorithm
{
public:
  MeshVelocityAlg(Realm&, stk::mesh::Part*);

  virtual ~MeshVelocityAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests elemData_;

  unsigned modelCoords_{stk::mesh::InvalidOrdinal};
  unsigned currentCoords_{stk::mesh::InvalidOrdinal};
  unsigned meshDispNp1_{stk::mesh::InvalidOrdinal};
  unsigned meshDispN_{stk::mesh::InvalidOrdinal};
  unsigned faceVelMag_{stk::mesh::InvalidOrdinal};
  unsigned sweptVolumeNp1_{stk::mesh::InvalidOrdinal};
  unsigned sweptVolumeN_{stk::mesh::InvalidOrdinal};

  MasterElement* meSCS_{nullptr};

  const double isoParCoords_[57] = {
    0.00,  -1.00, -1.00, // surf 1    1->2  0  8
    1.00,  0.00,  -1.00, // surf 2    2->3  1  9
    0.00,  1.00,  -1.00, // surf 3    3->4  2 10
    -1.00, 0.00,  -1.00, // surf 4    1->4  3 11
    0.00,  0.00,  -1.00, //                 4 12
    0.00,  -1.00, 1.00,  // surf 5    5->6  5 13
    1.00,  0.00,  1.00,  // surf 6    6->7  6 14
    0.00,  1.00,  1.00,  // surf 7    7->8  7 15
    -1.00, 0.00,  1.00,  // surf 8    5->8  8 16
    0.00,  0.00,  1.00,  //                 9 17
    1.00,  -1.00, 0.00,  // surf 10   2->6 10 18
    -1.00, -1.00, 0.00,  // surf 9    1->5 11 19
    0.00,  -1.00, 0.00,  //                12 20
    -1.00, 1.00,  0.00,  // surf 12   4->8 14 21
    1.00,  1.00,  0.00,  // surf 11   3->7 13 22
    0.00,  1.00,  0.00,  //                15 23
    1.00,  0.00,  0.00,  //                16 24
    -1.00, 0.00,  0.00,  //                17 25
    0.00,  0.00,  0.00,  //                18 26
  };

  const int scsFaceNodeMap_[12][4] = {
    {12, 0, 4, 18},   {16, 1, 4, 18},   {2, 4, 18, 15},   {3, 17, 18, 4},
    {5, 12, 18, 9},   {9, 6, 16, 18},   {9, 7, 15, 18},   {8, 9, 18, 17},
    {11, 12, 18, 17}, {12, 10, 16, 18}, {14, 15, 18, 16}, {13, 17, 18, 15}};

  Kokkos::View<double*, sierra::nalu::MemSpace> isoCoordsShapeFcn_;
};

} // namespace nalu
} // namespace sierra

#endif /* MESHVELOCITYALG_H */
