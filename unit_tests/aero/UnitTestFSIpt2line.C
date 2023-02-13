// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <KokkosInterface.h>
#include <aero/fsi/FSIturbine.h>

namespace {

double
call_projectPt2Line_on_device(
  const sierra::nalu::Point& pt,
  const sierra::nalu::Point& lStart,
  const sierra::nalu::Point& lEnd)
{
  double result = 0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const unsigned& i, double& localResult) {
      localResult = sierra::nalu::projectPt2Line(pt, lStart, lEnd);
    },
    result);

  return result;
}

void
test_projectPt2Line()
{
  sierra::nalu::Point pt(0.5, 0.5, 0.5);
  sierra::nalu::Point lStart(0.0, 0.0, 0.0);
  sierra::nalu::Point lEnd(1.0, 1.0, 1.0);

  {
    double result = call_projectPt2Line_on_device(pt, lStart, lEnd);

    const double expectedResult = 0.5;
    EXPECT_NEAR(expectedResult, result, 1.e-12);
  }

  pt = sierra::nalu::Point(0.0, 0.0, 0.5);
  {
    double result = call_projectPt2Line_on_device(pt, lStart, lEnd);

    const double expectedResult = 0.16666667;
    EXPECT_NEAR(expectedResult, result, 1.e-6);
  }
}

TEST(FSI, projectPt2Line) { test_projectPt2Line(); }

double
call_perpProjectDist_Pt2Line_on_device(
  const sierra::nalu::Point& pt,
  const sierra::nalu::Point& lStart,
  const sierra::nalu::Point& lEnd)
{
  double result = 0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const unsigned& i, double& localResult) {
      localResult = sierra::nalu::perpProjectDist_Pt2Line(pt, lStart, lEnd);
    },
    result);

  return result;
}

void
test_perpProjectDist_Pt2Line()
{
  sierra::nalu::Point pt(0.5, 0.5, 0.5);
  sierra::nalu::Point lStart(0.0, 0.0, 0.0);
  sierra::nalu::Point lEnd(1.0, 1.0, 1.0);

  {
    double result = call_perpProjectDist_Pt2Line_on_device(pt, lStart, lEnd);

    const double expectedResult = 0.0;
    EXPECT_NEAR(expectedResult, result, 1.e-12);
  }

  pt = sierra::nalu::Point(0.0, 0.0, 0.5);

  {
    double result = call_perpProjectDist_Pt2Line_on_device(pt, lStart, lEnd);

    const double expectedResult = 0.23570226;
    EXPECT_NEAR(expectedResult, result, 1.e-6);
  }
}

TEST(FSI, perpProjectDist_Pt2Line) { test_perpProjectDist_Pt2Line(); }

} // namespace
