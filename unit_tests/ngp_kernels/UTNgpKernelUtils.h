/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "ElemDataRequestsGPU.h"
#include "ScratchViews.h"
#include "kernel/Kernel.h"
#include "utils/StkHelpers.h"

namespace unit_test_ngp_kernels {

template <typename AlgTraits>
class TestContinuityKernel
  : public sierra::nalu::NGPKernel<TestContinuityKernel<AlgTraits>>
{
public:
  using TeamType = sierra::nalu::DeviceTeamHandleType;
  using ShmemType = sierra::nalu::DeviceShmem;

  KOKKOS_FUNCTION
  TestContinuityKernel() = default;

  KOKKOS_FUNCTION
  virtual ~TestContinuityKernel() = default;

  TestContinuityKernel(
    const stk::mesh::BulkData& bulk,
    sierra::nalu::ElemDataRequests& dataReq)
  {
    auto& meta = bulk.mesh_meta_data();

    coordinates_ = sierra::nalu::get_field_ordinal(meta, "coordinates");
    velocity_    = sierra::nalu::get_field_ordinal(meta, "velocity");
    pressure_    = sierra::nalu::get_field_ordinal(meta, "pressure");

    auto* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element(stk::topology::HEX_8);

    dataReq.add_cvfem_surface_me(meSCS);

    dataReq.add_coordinates_field(coordinates_, AlgTraits::nDim_,
                                  sierra::nalu::CURRENT_COORDINATES);
    dataReq.add_gathered_nodal_field(velocity_, AlgTraits::nDim_);
    dataReq.add_gathered_nodal_field(pressure_, 1);
  }

  using sierra::nalu::Kernel::execute;
  void execute(
    sierra::nalu::SharedMemView<DoubleType**>&,
    sierra::nalu::SharedMemView<DoubleType*>&,
    sierra::nalu::ScratchViews<DoubleType>&);

  KOKKOS_FUNCTION
  virtual void execute(
    sierra::nalu::SharedMemView<DoubleType**, ShmemType>&,
    sierra::nalu::SharedMemView<DoubleType*, ShmemType>&,
    sierra::nalu::ScratchViews<double, TeamType, ShmemType>&);

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocity_    {stk::mesh::InvalidOrdinal};
  unsigned pressure_    {stk::mesh::InvalidOrdinal};
};

template<typename AlgTraits>
void
TestContinuityKernel<AlgTraits>::execute(
  sierra::nalu::SharedMemView<DoubleType**>&,
  sierra::nalu::SharedMemView<DoubleType*>& rhs,
  sierra::nalu::ScratchViews<DoubleType>& scratchViews)
{
  auto& v_velocity = scratchViews.get_scratch_view_2D(velocity_);
  auto& v_pressure = scratchViews.get_scratch_view_1D(pressure_);

  rhs(0) = v_velocity(0, 0) + v_pressure(0);
}

template<typename AlgTraits>
void
TestContinuityKernel<AlgTraits>::execute(
  sierra::nalu::SharedMemView<DoubleType**, ShmemType>&,
  sierra::nalu::SharedMemView<DoubleType*, ShmemType>& rhs,
  sierra::nalu::ScratchViews<double, TeamType, ShmemType>& scratchViews)
{
  auto& v_velocity = scratchViews.get_scratch_view_2D(velocity_);
  auto& v_pressure = scratchViews.get_scratch_view_1D(pressure_);

  rhs(0) = v_velocity(0, 0) + v_pressure(0);
}

} // unit_test_ngp_kernels
