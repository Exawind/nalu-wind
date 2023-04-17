// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef _BEAM_H_
#define _BEAM_H_

#include <vs/vector.h>
#include <KokkosInterface.h>
#include <aero/aero_utils/WienerMilenkovic.h>
#include <aero/actuator/ActuatorTypes.h>

namespace aero {

template <typename BEAM_MEM_SPACE>
class Beam
{
public:
  Beam() : m_refPos(0) {}

  Beam(size_t numPoints)
    : m_refPos("Beam", numPoints),
      m_def("Beam", numPoints),
      m_load("Beam", numPoints)
  {
  }

  KOKKOS_FUNCTION size_t size() const { return m_refPos.size(); }

  SixDOF& get_ref_pos(size_t pt) { return m_refPos(pt); }
  const SixDOF& get_ref_pos(size_t pt) const { return m_refPos(pt); }

  SixDOF& get_def(size_t pt) { return m_def(pt); }
  const SixDOF& get_def(size_t pt) const { return m_def(pt); }

  SixDOF& get_load(size_t pt) { return m_load(pt); }
  const SixDOF& get_load(size_t pt) const { return m_load(pt); }

  void copy_from(
    const std::vector<double>& refPos,
    const std::vector<double>& def,
    const std::vector<double>& load)
  {
    const size_t numPoints = refPos.size() / 6;
    m_refPos = Kokkos::View<SixDOF*, BEAM_MEM_SPACE>("Beam", numPoints);
    m_def = Kokkos::View<SixDOF*, BEAM_MEM_SPACE>("Beam", numPoints);
    m_load = Kokkos::View<SixDOF*, BEAM_MEM_SPACE>("Beam", numPoints);

    for (size_t pt = 0; pt < numPoints; ++pt) {
      for (int d = 0; d < 6; ++d) {
        m_refPos(pt)[d] = refPos[pt * 6 + d];
        m_def(pt)[d] = def[pt * 6 + d];
        m_load(pt)[d] = load[pt * 6 + d];
      }
    }
  }

  void copy_to(
    std::vector<double>& refPos,
    std::vector<double>& def,
    std::vector<double>& load)
  {
    const size_t numPoints = size();
    const size_t vecSize = numPoints * 6;
    refPos.resize(vecSize);
    def.resize(vecSize);
    load.resize(vecSize);

    for (size_t pt = 0; pt < numPoints; ++pt) {
      for (int d = 0; d < 6; ++d) {
        refPos[pt * 6 + d] = m_refPos(pt)[d];
        def[pt * 6 + d] = m_def(pt)[d];
        load[pt * 6 + d] = m_load(pt)[d];
      }
    }
  }

  template <typename ViewType>
  void deep_copy_from(const ViewType& srcView)
  {
    m_refPos = Kokkos::View<SixDOF*, BEAM_MEM_SPACE>("Beam", srcView.size());
    m_def = Kokkos::View<SixDOF*, BEAM_MEM_SPACE>("Beam", srcView.size());
    m_load = Kokkos::View<SixDOF*, BEAM_MEM_SPACE>("Beam", srcView.size());
    Kokkos::deep_copy(m_refPos, srcView.m_refPos);
    Kokkos::deep_copy(m_def, srcView.m_def);
    Kokkos::deep_copy(m_load, srcView.m_load);
  }

private:
  Kokkos::View<SixDOF*, BEAM_MEM_SPACE> m_refPos;
  Kokkos::View<SixDOF*, BEAM_MEM_SPACE> m_def;
  Kokkos::View<SixDOF*, BEAM_MEM_SPACE> m_load;
};

using HostBeam = Beam<sierra::nalu::ActuatorFixedMemSpace>;
using DeviceBeam = Beam<sierra::nalu::ActuatorMemSpace>;

} // namespace aero

#endif
