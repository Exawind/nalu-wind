// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef INTEGRATION_RULES_H
#define INTEGRATION_RULES_H

#include "AlgTraits.h"
#include "ArrayND.h"
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include "ElementBasis.h"

namespace sierra::nalu {

enum class QuadType { MID, SHIFTED };

template <QuadType>
struct EdgeIntegrationRule
{
};

template <>
struct EdgeIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[1][1]> scs{{{0}}};
  static constexpr ArrayND<double[2][1]> scv{{{-0.5}, {0.5}}};
};

template <>
struct EdgeIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[1][1]> scs{{{0}}};
  static constexpr ArrayND<double[2][1]> scv{{{-1}, {+1}}};
};

template <QuadType>
struct TriIntegrationRule
{
};

template <>
struct TriIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[3][2]> scs{
    {{5. / 12., 2. / 12.},   // surf 1: 0->1
     {5. / 12., 5. / 12.},   // surf 2: 1->3
     {2. / 12., 5. / 12.}}}; // surf 3: 0->2

  static constexpr ArrayND<double[3][2]> scv{
    {{5. / 24., 5. / 24.}, {7. / 12., 5. / 24.}, {5. / 24., 7. / 12.}}};
};

template <>
struct TriIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[3][2]> scs{
    {{0.5, 0.0},   // surf 1: 0->1
     {0.5, 0.5},   // surf 2: 1->3
     {0.0, 0.5}}}; // surf 3: 0->2

  static constexpr ArrayND<double[3][2]> scv{{{0, 0}, {1, 0}, {0, 1}}};
};

template <QuadType>
struct QuadIntegrationRule
{
};

template <>
struct QuadIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[4][2]> scs{
    {{+0.0, -0.5},   // surf 1; 1->2
     {+0.5, +0.0},   // surf 2; 2->3
     {+0.0, +0.5},   // surf 3; 3->4
     {-0.5, +0.0}}}; // surf 3; 1->5};

  static constexpr ArrayND<double[4][2]> scv{
    {{-0.5, -0.5}, {+0.5, -0.5}, {+0.5, +0.5}, {-0.5, +0.5}}};
};

template <>
struct QuadIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[4][2]> scs{
    {{+0, -1}, {+1, +0}, {+0, +1}, {-1, +0}}};

  static constexpr ArrayND<double[4][2]> scv{
    {{-1, -1}, {+1, -1}, {+1, +1}, {-1, +1}}};
};

template <QuadType>
struct HexIntegrationRule
{
};

template <>
struct HexIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[12][3]> scs{
    {{+0.0, -0.5, -0.5},   // surf 1    1->2
     {+0.5, +0.0, -0.5},   // surf 2    2->3
     {+0.0, +0.5, -0.5},   // surf 3    3->4
     {-0.5, +0.0, -0.5},   // surf 4    1->4
     {+0.0, -0.5, +0.5},   // surf 5    5->6
     {+0.5, +0.0, +0.5},   // surf 6    6->7
     {+0.0, +0.5, +0.5},   // surf 7    7->8
     {-0.5, +0.0, +0.5},   // surf 8    5->8
     {-0.5, -0.5, +0.0},   // surf 9    1->5
     {+0.5, -0.5, +0.0},   // surf 10   2->6
     {+0.5, +0.5, +0.0},   // surf 11   3->7
     {-0.5, +0.5, +0.0}}}; // surf 12   4->8

  static constexpr ArrayND<double[8][3]> scv{
    {{-0.5, -0.5, -0.5},
     {+0.5, -0.5, -0.5},
     {+0.5, +0.5, -0.5},
     {-0.5, +0.5, -0.5},
     {-0.5, -0.5, +0.5},
     {+0.5, -0.5, +0.5},
     {+0.5, +0.5, +0.5},
     {-0.5, +0.5, +0.5}}};
};

template <>
struct HexIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[12][3]> scs{
    {{+0, -1, -1},   // surf 1    1->2
     {+1, +0, -1},   // surf 2    2->3
     {+0, +1, -1},   // surf 3    3->4
     {-1, +0, -1},   // surf 4    1->4
     {+0, -1, +1},   // surf 5    5->6
     {+1, +0, +1},   // surf 6    6->7
     {+0, +1, +1},   // surf 7    7->8
     {-1, +0, +1},   // surf 8    5->8
     {-1, -1, +0},   // surf 9    1->5
     {+1, -1, +0},   // surf 10   2->6
     {+1, +1, +0},   // surf 11   3->7
     {-1, +1, +0}}}; // surf 12   4->8

  static constexpr ArrayND<double[8][3]> scv{
    {{-1, -1, -1},
     {+1, -1, -1},
     {+1, +1, -1},
     {-1, +1, -1},
     {-1, -1, +1},
     {+1, -1, +1},
     {+1, +1, +1},
     {-1, +1, +1}}};
};

template <QuadType>
struct TetIntegrationRule
{
};

template <>
struct TetIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[6][3]> scs{
    {{13. / 36., 05. / 36., 05. / 36.}, // surf 1    1->2
     {13. / 36., 13. / 36., 05. / 36.}, // surf 2    2->3
     {05. / 36., 13. / 36., 05. / 36.}, // surf 3    1->3
     {05. / 36., 05. / 36., 13. / 36.}, // surf 4    1->4
     {13. / 36., 05. / 36., 13. / 36.}, // surf 5    2->4
     {05. / 36., 13. / 36., 13. / 36.}}};

  static constexpr ArrayND<double[4][3]> scv{
    {{17. / 96., 17. / 96., 17. / 96.},
     {45. / 96., 17. / 96., 17. / 96.},
     {17. / 96., 45. / 96., 17. / 96.},
     {17. / 96., 17. / 96., 45. / 96.}}};
};

template <>
struct TetIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[6][3]> scs{{
    {0.5, 0.0, 0.0}, // surf 1    1->2
    {0.5, 0.5, 0.0}, // surf 2    2->3
    {0.0, 0.5, 0.0}, // surf 3    1->3
    {0.0, 0.0, 0.5}, // surf 4    1->4
    {0.5, 0.0, 0.5}, // surf 5    2->4
    {0.0, 0.5, 0.5}  // surf 6    3->4
  }};

  static constexpr ArrayND<double[4][3]> scv{
    {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
};

template <QuadType>
struct WedIntegrationRule
{
};

template <>
struct WedIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[9][3]> scs{
    {{+05. / 12., +01. / 06., -01. / 02.},
     {+05. / 12., +05. / 12., -01. / 02.},
     {+01. / 06., +05. / 12., -01. / 02.},
     {+05. / 12., +01. / 06., +01. / 02.},
     {+05. / 12., +05. / 12., +01. / 02.},
     {+01. / 06., +05. / 12., +01. / 02.},
     {+07. / 36., +07. / 36., +00. / 02.},
     {+11. / 18., +07. / 36., +00. / 02.},
     {+07. / 36., +11. / 18., +00. / 02.}}};

  static constexpr ArrayND<double[6][3]> scv{
    {{+07. / 36., +07. / 36., -01. / 02.},
     {+11. / 18., +07. / 36., -01. / 02.},
     {+07. / 36., +11. / 18., -01. / 02.},
     {+07. / 36., +07. / 36., +01. / 02.},
     {+11. / 18., +07. / 36., +01. / 02.},
     {+07. / 36., +11. / 18., +01. / 02.}}};
};

template <>
struct WedIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[9][3]> scs{
    {{+0.5, +0.0, -1.0},
     {+0.5, +0.5, -1.0},
     {+0.0, +0.5, -1.0},
     {+0.5, +0.0, +1.0},
     {+0.5, +0.5, +1.0},
     {+0.0, +0.5, +1.0},
     {+0.0, +0.0, +0.0},
     {+1.0, +0.0, +0.0},
     {+0.0, +1.0, +0.0}}};

  static constexpr ArrayND<double[6][3]> scv{
    {{+0, +0, -1},
     {+1, +0, -1},
     {+0, +1, -1},
     {+0, +0, +1},
     {+1, +0, +1},
     {+0, +1, +1}}};
};

template <QuadType>
struct PyrIntegrationRule
{
};

template <>
struct PyrIntegrationRule<QuadType::MID>
{
  static constexpr ArrayND<double[12][3]> scs{{
    {+000. / 315., -145. / 315., +041. / 315.}, // surf 0  1->2
    {+145. / 315., +000. / 315., +041. / 315.}, // surf 1  2->3
    {+000. / 315., +145. / 315., +041. / 315.}, // surf 2  3->4
    {-145. / 315., +000. / 315., +041. / 315.}, // surf 3  1->4
    {-070. / 315., -070. / 315., +091. / 315.}, // surf 4  1->5 inner
    {-007. / 018., -007. / 018., +007. / 018.}, // surf 5  1->5 outer
    {+070. / 315., -070. / 315., +091. / 315.}, // surf 6  2->5 inner
    {+007. / 018., -007. / 018., +007. / 018.}, // surf 7  2->5 outer
    {+070. / 315., +070. / 315., +091. / 315.}, // surf 8  3->5 inner
    {+007. / 018., +007. / 018., +007. / 018.}, // surf 9  3->5 outer
    {-070. / 315., +070. / 315., +091. / 315.}, // surf 10  4->5 inner
    {-007. / 018., +007. / 018., +007. / 018.}  // surf 11  4->5 outer
  }};

  static constexpr double one69r384 = 169.0 / 384.0;
  static constexpr double five77r3840 = 577.0 / 3840.0;
  static constexpr double seven73r1560 = 773.0 / 1560.0;
  static constexpr ArrayND<double[5][3]> scv{
    {{-one69r384, -one69r384, five77r3840},
     {one69r384, -one69r384, five77r3840},
     {one69r384, one69r384, five77r3840},
     {-one69r384, one69r384, five77r3840},
     {0.0, 0.0, seven73r1560}}};
};

template <>
struct PyrIntegrationRule<QuadType::SHIFTED>
{
  static constexpr ArrayND<double[12][3]> scs{{
    {+0.0, -1.0, +0.0}, // surf 1    1->2
    {+1.0, +0.0, +0.0}, // surf 2    2->3
    {+0.0, +1.0, +0.0}, // surf 3    3->4
    {-1.0, +0.0, +0.0}, // surf 4    1->4
    {-0.5, -0.5, +0.5}, // surf 5    1->5 I
    {-0.5, -0.5, +0.5}, // surf 6    1->5 O
    {+0.5, -0.5, +0.5}, // surf 7    2->5 I
    {+0.5, -0.5, +0.5}, // surf 8    2->5 O
    {+0.5, +0.5, +0.5}, // surf 9    3->5 I
    {+0.5, +0.5, +0.5}, // surf 10   3->5 O
    {-0.5, +0.5, +0.5}, // surf 11   4->5 I
    {-0.5, +0.5, +0.5}  // surf 12   4->5 O
  }};

  static constexpr ArrayND<double[5][3]> scv{
    {{-1, -1, +0}, {+1, -1, +0}, {+1, +1, +0}, {-1, +1, +0}, {+0, +0, +1}}};
};

} // namespace sierra::nalu
#endif