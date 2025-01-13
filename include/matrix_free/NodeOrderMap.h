// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NODEORDERMAP_H
#define NODEORDERMAP_H

#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int>
struct StkNodeOrderMapping
{
};

template <>
struct StkNodeOrderMapping<1>
{
  using node_map_type = ArrayND<int[2][2][2]>;
  static constexpr node_map_type map = {{{{0, 1}, {3, 2}}, {{4, 5}, {7, 6}}}};
};

template <>
struct StkNodeOrderMapping<2>
{
  using node_map_type = ArrayND<int[3][3][3]>;
  static constexpr node_map_type map = {
    {{{0, 8, 1}, {11, 21, 9}, {3, 10, 2}},
     {{12, 25, 13}, {23, 20, 24}, {15, 26, 14}},
     {{4, 16, 5}, {19, 22, 17}, {7, 18, 6}}}};
};

template <>
struct StkNodeOrderMapping<3>
{
  using node_map_type = ArrayND<int[4][4][4]>;
  static constexpr node_map_type map = {
    {{{0, 8, 9, 1}, {15, 32, 34, 10}, {14, 33, 35, 11}, {3, 13, 12, 2}},
     {{16, 48, 49, 18}, {40, 56, 57, 44}, {42, 58, 59, 45}, {22, 53, 52, 20}},
     {{17, 50, 51, 19}, {41, 60, 61, 46}, {43, 62, 63, 47}, {23, 55, 54, 21}},
     {{4, 24, 25, 5}, {31, 36, 37, 26}, {30, 38, 39, 27}, {7, 29, 28, 6}}}};
};

template <>
struct StkNodeOrderMapping<4>
{
  using node_map_type = ArrayND<int[5][5][5]>;
  static constexpr node_map_type map = {
    {{{0, 8, 9, 10, 1},
      {19, 44, 47, 50, 11},
      {18, 45, 48, 51, 12},
      {17, 46, 49, 52, 13},
      {3, 16, 15, 14, 2}},
     {{20, 80, 81, 82, 23},
      {62, 98, 99, 100, 71},
      {65, 101, 102, 103, 72},
      {68, 104, 105, 106, 73},
      {29, 91, 90, 89, 26}},
     {{21, 83, 84, 85, 24},
      {63, 107, 108, 109, 74},
      {66, 110, 111, 112, 75},
      {69, 113, 114, 115, 76},
      {30, 94, 93, 92, 27}},
     {{22, 86, 87, 88, 25},
      {64, 116, 117, 118, 77},
      {67, 119, 120, 121, 78},
      {70, 122, 123, 124, 79},
      {31, 97, 96, 95, 28}},
     {{4, 32, 33, 34, 5},
      {43, 53, 54, 55, 35},
      {42, 56, 57, 58, 36},
      {41, 59, 60, 61, 37},
      {7, 40, 39, 38, 6}}}};
};

template <int>
struct StkFaceNodeMapping
{
};

template <>
struct StkFaceNodeMapping<1>
{
  using node_map_type = ArrayND<int[2][2]>;
  static constexpr node_map_type map = {{{0, 1}, {3, 2}}};
};

template <>
struct StkFaceNodeMapping<2>
{
  using node_map_type = ArrayND<int[3][3]>;
  static constexpr node_map_type map = {{{0, 4, 1}, {7, 8, 5}, {3, 6, 2}}};
};

template <>
struct StkFaceNodeMapping<3>
{
  using node_map_type = ArrayND<int[4][4]>;
  static constexpr node_map_type map = {
    {{0, 4, 5, 1}, {11, 12, 13, 6}, {10, 14, 15, 7}, {3, 9, 8, 2}}};
};

template <>
struct StkFaceNodeMapping<4>
{
  using node_map_type = ArrayND<int[5][5]>;
  static constexpr node_map_type map = {
    {{0, 4, 5, 6, 1},
     {15, 16, 17, 18, 7},
     {14, 19, 20, 21, 8},
     {13, 22, 23, 24, 9},
     {3, 12, 11, 10, 2}}};
};

int node_map(int poly, int n, int m, int l);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
