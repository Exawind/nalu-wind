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

    meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();

    dataReq.add_cvfem_surface_me(meSCS_);

    dataReq.add_coordinates_field(coordinates_, AlgTraits::nDim_,
                                  sierra::nalu::CURRENT_COORDINATES);
    dataReq.add_gathered_nodal_field(velocity_, AlgTraits::nDim_);
    dataReq.add_gathered_nodal_field(pressure_, 1);
    dataReq.add_master_element_call(
      sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);
    dataReq.add_master_element_call(
      sierra::nalu::SCS_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);
  }

  using sierra::nalu::Kernel::execute;

  KOKKOS_FUNCTION
  void execute(
    sierra::nalu::SharedMemView<DoubleType**, ShmemType>&,
    sierra::nalu::SharedMemView<DoubleType*, ShmemType>&,
    sierra::nalu::ScratchViews<DoubleType, TeamType, ShmemType>&);

  KOKKOS_FUNCTION
  virtual void execute(
    sierra::nalu::SharedMemView<DoubleType**, ShmemType>&,
    sierra::nalu::SharedMemView<DoubleType*, ShmemType>&,
    sierra::nalu::ScratchViews<double, TeamType, ShmemType>&);

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocity_    {stk::mesh::InvalidOrdinal};
  unsigned pressure_    {stk::mesh::InvalidOrdinal};

  sierra::nalu::MasterElement* meSCS_;
};

template<typename AlgTraits>
void
TestContinuityKernel<AlgTraits>::execute(
  sierra::nalu::SharedMemView<DoubleType**, ShmemType>&,
  sierra::nalu::SharedMemView<DoubleType*, ShmemType>& rhs,
  sierra::nalu::ScratchViews<DoubleType, TeamType, ShmemType>& scratchViews)
{
  // Get the integration point to node mapping
  const int* ipNodeMap = meSCS_->ipNodeMap(3);

  auto& v_velocity = scratchViews.get_scratch_view_2D(velocity_);
  auto& v_pressure = scratchViews.get_scratch_view_1D(pressure_);

  auto& meViews = scratchViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);
  auto& scs_areav = meViews.scs_areav;
  auto& v_shape_fcn = meViews.scs_shape_fcn;

  printf("ipNodeMap[2] = %d (7); SCS areav[2, 0] = %f\n", ipNodeMap[2],
         stk::simd::get_data(scs_areav(2, 0), 0));

  rhs(0) = v_velocity(0, 0) + v_pressure(0) * v_shape_fcn(0, 0);
}

template<typename AlgTraits>
void
TestContinuityKernel<AlgTraits>::execute(
  sierra::nalu::SharedMemView<DoubleType**, ShmemType>&,
  sierra::nalu::SharedMemView<DoubleType*, ShmemType>& rhs,
  sierra::nalu::ScratchViews<double, TeamType, ShmemType>& scratchViews)
{
  // Get the integration point to node mapping
  const int* ipNodeMap = meSCS_->ipNodeMap(3);

  auto& v_velocity = scratchViews.get_scratch_view_2D(velocity_);
  auto& v_pressure = scratchViews.get_scratch_view_1D(pressure_);

  rhs(0) = v_velocity(0, 0) + v_pressure(0);

  printf("ipNodeMap[2] = %d; expected = 7\n", ipNodeMap[2]);
}

} // unit_test_ngp_kernels
