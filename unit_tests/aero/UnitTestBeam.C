// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <vs/vector.h>
#include <aero/aero_utils/displacements.h>
#include <aero/aero_utils/Beam.h>

TEST(SixDOF, sizeof_same_as_6_doubles)
{
  EXPECT_EQ(sizeof(double) * 6, sizeof(aero::SixDOF));
}

TEST(HostBeam, basic_construction)
{
  const unsigned numPoints = 10;
  aero::HostBeam beam(numPoints);
  EXPECT_EQ(numPoints, beam.size());
}

void
fill_vecs(
  std::vector<double>& refPos,
  std::vector<double>& def,
  std::vector<double>& load)
{
  double refPosData = 1.0;
  double defData = 9.9;
  double loadData = 42.0;
  for (unsigned i = 0; i < refPos.size(); ++i) {
    refPos[i] = refPosData + i * 1.0;
    def[i] = defData + i * 1.1;
    load[i] = loadData + i * 1.2;
  }
}

template <typename BeamType>
void
verify_equal(
  const BeamType& beam,
  const std::vector<double>& refPos,
  const std::vector<double>& def,
  const std::vector<double>& load)
{
  const unsigned numPoints = beam.size();
  for (unsigned pt = 0; pt < numPoints; ++pt) {
    const aero::SixDOF& beamRefPos = beam.get_ref_pos(pt);
    const aero::SixDOF& beamDef = beam.get_def(pt);
    const aero::SixDOF& beamLoad = beam.get_load(pt);
    for (unsigned dof = 0; dof < 6; ++dof) {
      EXPECT_NEAR(refPos[pt * 6 + dof], beamRefPos[dof], 1.e-6);
      EXPECT_NEAR(def[pt * 6 + dof], beamDef[dof], 1.e-6);
      EXPECT_NEAR(load[pt * 6 + dof], beamLoad[dof], 1.e-6);
    }
  }
}

TEST(HostBeam, copy_from_vectors)
{
  const unsigned numPoints = 10;
  aero::HostBeam beam;

  std::vector<double> refPos(numPoints * 6);
  std::vector<double> def(numPoints * 6);
  std::vector<double> load(numPoints * 6);
  fill_vecs(refPos, def, load);

  beam.copy_from(refPos, def, load);
  EXPECT_EQ(numPoints, beam.size());

  verify_equal(beam, refPos, def, load);
}

TEST(HostBeam, copy_to_vector)
{
  const unsigned numPoints = 10;
  aero::HostBeam beam;

  std::vector<double> gold_refPos(numPoints * 6);
  std::vector<double> gold_def(numPoints * 6);
  std::vector<double> gold_load(numPoints * 6);
  fill_vecs(gold_refPos, gold_def, gold_load);

  beam.copy_from(gold_refPos, gold_def, gold_load);

  std::vector<double> refPos(numPoints * 6);
  std::vector<double> def(numPoints * 6);
  std::vector<double> load(numPoints * 6);
  beam.copy_to(refPos, def, load);

  verify_equal(beam, refPos, def, load);
}

void
check_size_on_device(int expectedSize, const aero::DeviceBeam& d_beam)
{
  int sizeFromDevice = 0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const unsigned& i, int& localSize) {
      localSize = d_beam.size();
    },
    sizeFromDevice);

  EXPECT_EQ(expectedSize, sizeFromDevice);
}

TEST(HostBeam, copy_to_device)
{
  const unsigned numPoints = 10;
  aero::HostBeam h_beam(numPoints);
  aero::DeviceBeam d_beam;
  d_beam.deep_copy_from(h_beam);

  check_size_on_device(numPoints, d_beam);
}
