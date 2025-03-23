// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "UnitTestUtils.h"
#include "kernels/UnitTestKernelUtils.h"
#include "master_element/MasterElement.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpReducers.h"
#include "master_element/Hex8CVFEM.h"
#include "master_element/Quad43DCVFEM.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "Kokkos_DualView.hpp"

#include <cmath>

class NgpLoopTest : public ::testing::Test
{
public:
  NgpLoopTest() : partVec()
  {
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(3);
    bulk = meshBuilder.create();
    meta = &bulk->mesh_meta_data();
    meta->use_simple_fields();

    density = &meta->declare_field<double>(stk::topology::NODE_RANK, "density");
    pressure =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "pressure");
    velocity =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "velocity");
    mdotEdge =
      &meta->declare_field<double>(stk::topology::EDGE_RANK, "mass_flow_rate");
    massFlowRate = &meta->declare_field<double>(
      stk::topology::ELEM_RANK, "mass_flow_rate_scs");

    const double ten = 10.0;
    const double zero = 0.0;
    const double oneVec[3] = {1.0, 1.0, 1.0};
    sierra::nalu::HexSCS hex8SCS;
    stk::mesh::put_field_on_mesh(*density, meta->universal_part(), &ten);
    stk::mesh::put_field_on_mesh(*pressure, meta->universal_part(), &zero);
    stk::mesh::put_field_on_mesh(*velocity, meta->universal_part(), 3, oneVec);
    stk::io::set_field_output_type(
      *velocity, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *massFlowRate, meta->universal_part(), hex8SCS.num_integration_points(),
      nullptr);
    stk::mesh::put_field_on_mesh(*mdotEdge, meta->universal_part(), &zero);
  }

  ~NgpLoopTest() = default;

  void
  fill_mesh_and_init_fields(const std::string& meshSpec = "generated:2x2x2")
  {
    unit_test_utils::fill_hex8_mesh(meshSpec, *bulk);
    partVec = {meta->get_part("block_1")};

    coordField = static_cast<const sierra::nalu::VectorFieldType*>(
      meta->coordinate_field());
    EXPECT_TRUE(coordField != nullptr);
  }

  stk::mesh::MetaData* meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::PartVector partVec;
  const sierra::nalu::VectorFieldType* coordField{nullptr};
  sierra::nalu::ScalarFieldType* density{nullptr};
  sierra::nalu::ScalarFieldType* pressure{nullptr};
  sierra::nalu::VectorFieldType* velocity{nullptr};
  sierra::nalu::ScalarFieldType* mdotEdge{nullptr};
  sierra::nalu::GenericFieldType* massFlowRate{nullptr};
};

void
basic_node_loop(
  const stk::mesh::BulkData& bulk, sierra::nalu::ScalarFieldType& pressure)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  const double presSet = 4.0;

  const auto& meta = bulk.mesh_meta_data();
  stk::mesh::Selector sel = meta.universal_part();
  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpPressure =
    stk::mesh::get_updated_ngp_field<double>(pressure);

  sierra::nalu::nalu_ngp::run_entity_algorithm(
    "unittest_basic_node_loop", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& meshIdx) {
      ngpPressure.get(meshIdx, 0) = presSet;
    });

  ngpPressure.modify_on_device();
  ngpPressure.sync_to_host();

  // Do checks
  {
    const auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);
    const double tol = 1.0e-16;
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        const double* pres = stk::mesh::field_data(pressure, node);
        EXPECT_NEAR(presSet, pres[0], tol);
      }
    }
  }
}

void
basic_node_reduce(
  const stk::mesh::BulkData& bulk, sierra::nalu::ScalarFieldType& pressure)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  const double presSet = 4.0;

  stk::mesh::field_fill(presSet, pressure);
  const auto& meta = bulk.mesh_meta_data();
  stk::mesh::Selector sel = meta.universal_part();
  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpPressure =
    stk::mesh::get_updated_ngp_field<double>(pressure);

  double reduceVal = 0.0;
  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "unittest_basic_node_reduce1", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi, double& pSum) {
      pSum += ngpPressure.get(mi, 0);
    },
    reduceVal);

  double reduceVal1 = 0.0;
  Kokkos::Sum<double> sum_reducer(reduceVal1);
  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "unittest_basic_node_reduce2", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi, double& pSum) {
      sum_reducer.join(pSum, ngpPressure.get(mi, 0));
    },
    sum_reducer);

  {
    double expectedSum = 0.0;
    const auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);
    const double tol = 1.0e-16;
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        const double* pres = stk::mesh::field_data(pressure, node);
        expectedSum += pres[0];
      }
    }
    EXPECT_NEAR(reduceVal, expectedSum, tol);
    EXPECT_NEAR(reduceVal1, expectedSum, tol);
  }
}

void
basic_node_reduce_minmax(
  const stk::mesh::BulkData& bulk, const double minGold, const double maxGold)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = bulk.mesh_meta_data();
  const auto& coords = meta.coordinate_field();
  stk::mesh::Selector sel = meta.universal_part();
  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpCoords =
    stk::mesh::get_updated_ngp_field<double>(*coords);

  using value_type = Kokkos::Max<double>::value_type;
  value_type max;
  Kokkos::Max<double> max_reducer(max);
  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "unittest_basic_node_reduce_max", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi, value_type& pSum) {
      const double xcoord = ngpCoords.get(mi, 0);
      if (xcoord > pSum)
        pSum = xcoord;
    },
    max_reducer);

  using value_type = Kokkos::Min<double>::value_type;
  value_type min;
  Kokkos::Min<double> min_reducer(min);
  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "unittest_basic_node_reduce_min", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi, value_type& pSum) {
      const double xcoord = ngpCoords.get(mi, 0);
      if (xcoord < pSum)
        pSum = xcoord;
    },
    min_reducer);

  EXPECT_NEAR(max, maxGold, tol);
  EXPECT_NEAR(min, minGold, tol);
}

void
basic_node_reduce_minmax_alt(
  const stk::mesh::BulkData& bulk, const double minGold, const double maxGold)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = bulk.mesh_meta_data();
  const auto& coords = meta.coordinate_field();
  stk::mesh::Selector sel = meta.universal_part();
  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpCoords =
    stk::mesh::get_updated_ngp_field<double>(*coords);

  using value_type = Kokkos::MinMax<double>::value_type;
  value_type minmax;
  Kokkos::MinMax<double> minmax_reducer(minmax);
  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "unittest_node_reduce_minmax", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi, value_type& threadVal) {
      const double xcoord = ngpCoords.get(mi, 0);
      if (xcoord < threadVal.min_val)
        threadVal.min_val = xcoord;
      if (xcoord > threadVal.max_val)
        threadVal.max_val = xcoord;
    },
    minmax_reducer);

  EXPECT_NEAR(minmax.max_val, maxGold, tol);
  EXPECT_NEAR(minmax.min_val, minGold, tol);
}

void
basic_node_reduce_minmaxsum(
  const stk::mesh::BulkData& bulk,
  const double minGold,
  const double maxGold,
  const double sumGold)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using MinMaxSum = sierra::nalu::nalu_ngp::MinMaxSum<double>;
  using value_type = typename MinMaxSum::value_type;

  const auto& meta = bulk.mesh_meta_data();
  const auto& coords = meta.coordinate_field();
  stk::mesh::Selector sel = meta.universal_part();
  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpCoords =
    stk::mesh::get_updated_ngp_field<double>(*coords);

  value_type minmaxsum;
  MinMaxSum reducer(minmaxsum);
  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "unittest_node_reduce_minmaxsum", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi, value_type& threadVal) {
      const double xcoord = ngpCoords.get(mi, 0);
      if (xcoord < threadVal.min_val)
        threadVal.min_val = xcoord;
      if (xcoord > threadVal.max_val)
        threadVal.max_val = xcoord;
      threadVal.total_sum += 1.0;
    },
    reducer);

  EXPECT_NEAR(minmaxsum.max_val, maxGold, tol);
  EXPECT_NEAR(minmaxsum.min_val, minGold, tol);
  EXPECT_NEAR(minmaxsum.total_sum, sumGold, tol);
}

void
basic_node_reduce_array(
  const stk::mesh::BulkData& bulk,
  sierra::nalu::ScalarFieldType& pressure,
  int num_nodes)
{
  using MeshIndex =
    sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  const double presSet = 4.0;

  stk::mesh::field_fill(presSet, pressure);
  const auto& meta = bulk.mesh_meta_data();
  stk::mesh::Selector sel = meta.universal_part();
  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpPressure =
    stk::mesh::get_updated_ngp_field<double>(pressure);

  using value_type = Kokkos::Sum<sierra::nalu::nalu_ngp::ArrayDbl2>::value_type;
  value_type lsum;
  Kokkos::Sum<sierra::nalu::nalu_ngp::ArrayDbl2> sum_reducer(lsum);

  sierra::nalu::nalu_ngp::run_entity_par_reduce(
    "basic_node_reduce_arrray", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi, value_type& pSum) {
      pSum.array_[0] += ngpPressure.get(mi, 0);
      pSum.array_[1] += 1.0;
    },
    sum_reducer);

  const double expectedPressureSum = presSet * num_nodes;
  const int computedNumNodes = static_cast<int>(lsum.array_[1]);
  EXPECT_NEAR(lsum.array_[0], expectedPressureSum, 1.0e-15);
  EXPECT_EQ(num_nodes, computedNumNodes);
}

void
basic_elem_loop(
  const stk::mesh::BulkData& bulk,
  sierra::nalu::ScalarFieldType& pressure,
  sierra::nalu::GenericFieldType& massFlowRate)
{
  const double flowRate = 4.0;
  const double presSet = 10.0;

  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpMassFlowRate =
    stk::mesh::get_updated_ngp_field<double>(massFlowRate);
  stk::mesh::NgpField<double>& ngpPressure =
    stk::mesh::get_updated_ngp_field<double>(pressure);

  const auto& meta = bulk.mesh_meta_data();
  stk::mesh::Selector sel = meta.universal_part();

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_basic_elem_loop", ngpMesh, stk::topology::ELEMENT_RANK, sel,
    KOKKOS_LAMBDA(
      const sierra::nalu::nalu_ngp::EntityInfo<stk::mesh::NgpMesh>& einfo) {
      ngpMassFlowRate.get(einfo.meshIdx, 0) = flowRate;

      const auto& nodes = einfo.entityNodes;
      const int numNodes = nodes.size();
      for (int i = 0; i < numNodes; ++i)
        ngpPressure.get(ngpMesh, nodes[i], 0) = presSet;
    });

  ngpMassFlowRate.modify_on_device();
  ngpMassFlowRate.sync_to_host();
  ngpPressure.modify_on_device();
  ngpPressure.sync_to_host();

  {
    const auto& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, sel);
    const double tol = 1.0e-16;
    for (const stk::mesh::Bucket* b : elemBuckets) {
      for (stk::mesh::Entity elem : *b) {
        const double* flowRateData = stk::mesh::field_data(massFlowRate, elem);
        EXPECT_NEAR(flowRate, *flowRateData, tol);

        const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
        const unsigned numNodes = bulk.num_nodes(elem);
        for (unsigned n = 0; n < numNodes; ++n) {
          const double* pres = stk::mesh::field_data(pressure, nodes[n]);
          EXPECT_NEAR(presSet, pres[0], tol);
        }
      }
    }
  }
}

void
basic_edge_loop(
  const stk::mesh::BulkData& bulk,
  sierra::nalu::ScalarFieldType& pressure,
  sierra::nalu::ScalarFieldType& mdotEdge)
{
  const double flowRate = 4.0;
  const double presSet = 10.0;

  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpMassFlowRate =
    stk::mesh::get_updated_ngp_field<double>(mdotEdge);
  stk::mesh::NgpField<double>& ngpPressure =
    stk::mesh::get_updated_ngp_field<double>(pressure);

  const auto& meta = bulk.mesh_meta_data();
  stk::mesh::Selector sel = meta.universal_part();

  sierra::nalu::nalu_ngp::run_edge_algorithm(
    "unittest_basic_edge_loop", ngpMesh, sel,
    KOKKOS_LAMBDA(
      const sierra::nalu::nalu_ngp::EntityInfo<stk::mesh::NgpMesh>& einfo) {
      ngpMassFlowRate.get(einfo.meshIdx, 0) = flowRate;

      const auto& nodes = einfo.entityNodes;
      const int numNodes = nodes.size();
      for (int i = 0; i < numNodes; ++i)
        ngpPressure.get(ngpMesh, nodes[i], 0) = presSet;
    });

  ngpMassFlowRate.modify_on_device();
  ngpMassFlowRate.sync_to_host();
  ngpPressure.modify_on_device();
  ngpPressure.sync_to_host();

  {
    const auto& edgeBuckets = bulk.get_buckets(stk::topology::EDGE_RANK, sel);
    const double tol = 1.0e-16;
    for (const stk::mesh::Bucket* b : edgeBuckets) {
      for (stk::mesh::Entity edge : *b) {
        const double* flowRateData = stk::mesh::field_data(mdotEdge, edge);
        EXPECT_NEAR(flowRate, *flowRateData, tol);

        const stk::mesh::Entity* nodes = bulk.begin_nodes(edge);
        const unsigned numNodes = bulk.num_nodes(edge);
        for (unsigned n = 0; n < numNodes; ++n) {
          const double* pres = stk::mesh::field_data(pressure, nodes[n]);
          EXPECT_NEAR(presSet, pres[0], tol);
        }
      }
    }
  }
}

void
elem_loop_scratch_views(
  const stk::mesh::BulkData& bulk,
  sierra::nalu::ScalarFieldType& pressure,
  sierra::nalu::VectorFieldType& velocity)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using Hex8Traits = sierra::nalu::AlgTraitsHex8;
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  typedef Kokkos::DualView<
    double*, Kokkos::LayoutRight, sierra::nalu::DeviceSpace>
    DoubleTypeView;

  const auto& meta = bulk.mesh_meta_data();
  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_dev(
      sierra::nalu::AlgTraitsHex8::topo_);
  dataReq.add_cvfem_volume_me(meSCV);

  auto* coordsField = bulk.mesh_meta_data().coordinate_field();
  dataReq.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(velocity, 3);
  dataReq.add_gathered_nodal_field(pressure, 1);
  dataReq.add_master_element_call(
    sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_master_element_call(
    sierra::nalu::SCV_SHIFTED_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  stk::mesh::Selector sel = meta.universal_part();

  const unsigned velID = velocity.mesh_meta_data_ordinal();
  const unsigned presID = pressure.mesh_meta_data_ordinal();

  // Field updates
  Traits::DblType xVel = 1.0;
  Traits::DblType yVel = 2.0;
  Traits::DblType zVel = 3.0;

  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpVel = fieldMgr.get_field<double>(velID);
  const auto ngpVelOp =
    sierra::nalu::nalu_ngp::simd_elem_nodal_field_updater(ngpMesh, ngpVel);

  const int numNodes = 8;
  DoubleTypeView volCheck("scv_volume", numNodes);
  Kokkos::deep_copy(volCheck.h_view, 0.0);
  volCheck.template modify<typename DoubleTypeView::host_mirror_space>();
  volCheck.template sync<typename DoubleTypeView::execution_space>();

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_elem_loop_scratchviews", meshInfo, stk::topology::ELEM_RANK,
    dataReq, sel, KOKKOS_LAMBDA(ElemSimdData & edata) {
      Traits::DblType test = 0.0;
      auto& scrViews = edata.simdScrView;
      auto& v_pres = scrViews.get_scratch_view_1D(presID);
      auto& v_vel = scrViews.get_scratch_view_2D(velID);
      auto& scv_vol =
        scrViews.get_me_views(sierra::nalu::CURRENT_COORDINATES).scv_volume;

      test += v_vel(0, 0) + v_pres(0) * scv_vol(0);

      for (int i = 0; i < numNodes; ++i) {
        volCheck.d_view(i) = stk::simd::get_data(scv_vol(i), 0);

        // Scatter SIMD value to nodes
        ngpVelOp(edata, i, 0) = xVel;
        ngpVelOp(edata, i, 1) = yVel;
        ngpVelOp(edata, i, 2) = zVel;
      }
    });

  ngpVel.modify_on_device();
  ngpVel.sync_to_host();

  volCheck.modify<DoubleTypeView::execution_space>();
  volCheck.sync<DoubleTypeView::host_mirror_space>();

  for (int i = 0; i < numNodes; ++i)
    EXPECT_NEAR(volCheck.h_view(i), 0.125, 1.0e-12);

  {
    const double xVel = 1.0;
    const double yVel = 2.0;
    const double zVel = 3.0;
    const auto& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, sel);
    const double tol = 1.0e-16;
    for (const stk::mesh::Bucket* b : elemBuckets) {
      for (stk::mesh::Entity elem : *b) {
        const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
        for (int i = 0; i < Hex8Traits::nodesPerElement_; ++i) {
          const double* velptr = stk::mesh::field_data(velocity, nodes[i]);
          EXPECT_NEAR(velptr[0], xVel, tol);
          EXPECT_NEAR(velptr[1], yVel, tol);
          EXPECT_NEAR(velptr[2], zVel, tol);
        }
      }
    }
  }
}

void
calc_mdot_elem_loop(
  const stk::mesh::BulkData& bulk,
  sierra::nalu::ScalarFieldType& density,
  sierra::nalu::VectorFieldType& velocity,
  sierra::nalu::GenericFieldType& massFlowRate)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using Hex8Traits = sierra::nalu::AlgTraitsHex8;
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meta = bulk.mesh_meta_data();
  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      Hex8Traits::topo_);
  dataReq.add_cvfem_surface_me(meSCS);

  auto* coordsField = bulk.mesh_meta_data().coordinate_field();
  dataReq.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(velocity, 3);
  dataReq.add_gathered_nodal_field(density, 1);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_SHIFTED_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  stk::mesh::Selector sel = meta.universal_part();

  const unsigned velID = velocity.mesh_meta_data_ordinal();
  const unsigned rhoID = density.mesh_meta_data_ordinal();
  const auto mdotID = massFlowRate.mesh_meta_data_ordinal();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  stk::mesh::NgpField<double> ngpMdot = fieldMgr.get_field<double>(mdotID);
  // SIMD Element field operation handler
  const auto mdotOps =
    sierra::nalu::nalu_ngp::simd_elem_field_updater(ngpMesh, ngpMdot);

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_calc_mdot_elem_loop", meshInfo, stk::topology::ELEM_RANK, dataReq,
    sel, KOKKOS_LAMBDA(ElemSimdData & edata) {
      NALU_ALIGNED Traits::DblType rhoU[Hex8Traits::nDim_];
      auto& scrViews = edata.simdScrView;
      auto& v_rho = scrViews.get_scratch_view_1D(rhoID);
      auto& v_vel = scrViews.get_scratch_view_2D(velID);
      auto& meViews = scrViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);
      auto& v_area = meViews.scs_areav;
      auto& v_shape_fcn = meViews.scs_shifted_shape_fcn;

      for (int ip = 0; ip < Hex8Traits::numScsIp_; ++ip) {
        for (int d = 0; d < Hex8Traits::nDim_; ++d)
          rhoU[d] = 0.0;

        for (int ic = 0; ic < Hex8Traits::nodesPerElement_; ++ic) {
          const auto r = v_shape_fcn(ip, ic);
          for (int d = 0; d < Hex8Traits::nDim_; ++d)
            rhoU[d] += r * v_rho(ic) * v_vel(ic, d);
        }

        Traits::DblType tmdot = 0.0;
        for (int d = 0; d < Hex8Traits::nDim_; ++d)
          tmdot += rhoU[d] * v_area(ip, d);

        mdotOps(edata, ip) = tmdot;
      }
    });

  ngpMdot.modify_on_device();
  ngpMdot.sync_to_host();

  {
    const double flowRate = 2.5;
    const auto& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, sel);
    const double tol = 1.0e-16;
    for (const stk::mesh::Bucket* b : elemBuckets) {
      for (stk::mesh::Entity elem : *b) {
        const double* flowRateData = stk::mesh::field_data(massFlowRate, elem);
        for (int i = 0; i < Hex8Traits::numScsIp_; ++i)
          EXPECT_NEAR(flowRate, std::abs(flowRateData[i]), tol);
      }
    }
  }
}

void
basic_face_elem_loop(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordField,
  const sierra::nalu::GenericFieldType& exposedArea,
  sierra::nalu::ScalarFieldType& wallArea,
  sierra::nalu::ScalarFieldType& wallNormDist)
{
  using MeshIndex =
    sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  using FaceTraits = sierra::nalu::AlgTraitsQuad4Hex8;
  using FaceSimdData =
    sierra::nalu::nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();

  sierra::nalu::ElemDataRequests faceData(meta);
  sierra::nalu::ElemDataRequests elemData(meta);
  auto meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      FaceTraits::FaceTraits::topo_);
  auto meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      FaceTraits::ElemTraits::topo_);

  faceData.add_cvfem_face_me(meFC);
  elemData.add_cvfem_surface_me(meSCS);
  faceData.add_coordinates_field(
    coordField, ndim, sierra::nalu::CURRENT_COORDINATES);
  faceData.add_face_field(
    exposedArea, FaceTraits::numFaceIp_, FaceTraits::nDim_);
  faceData.add_master_element_call(
    sierra::nalu::FC_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);
  elemData.add_coordinates_field(
    coordField, ndim, sierra::nalu::CURRENT_COORDINATES);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  stk::mesh::Part* part = meta.get_part("surface_5");
  stk::mesh::Selector sel(*part);

  const unsigned coordsID = coordField.mesh_meta_data_ordinal();
  const unsigned exposedAreaID = exposedArea.mesh_meta_data_ordinal();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto wArea = fieldMgr.get_field<double>(wallArea.mesh_meta_data_ordinal());
  auto wDist =
    fieldMgr.get_field<double>(wallNormDist.mesh_meta_data_ordinal());
  const auto areaOps =
    sierra::nalu::nalu_ngp::simd_face_elem_nodal_field_updater(ngpMesh, wArea);
  const auto distOps =
    sierra::nalu::nalu_ngp::simd_face_elem_nodal_field_updater(ngpMesh, wDist);

  sierra::nalu::nalu_ngp::run_face_elem_algorithm(
    "unittest_basic_face_elem_loop", meshInfo, faceData, elemData, sel,
    KOKKOS_LAMBDA(FaceSimdData & fdata) {
      auto& v_coord = fdata.simdElemView.get_scratch_view_2D(coordsID);
      auto& v_area = fdata.simdFaceView.get_scratch_view_2D(exposedAreaID);

      const int* faceIpNodeMap = meFC->ipNodeMap();
      for (int ip = 0; ip < FaceTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d = 0; d < FaceTraits::nDim_; ++d)
          aMag += v_area(ip, d) * v_area(ip, d);
        aMag = stk::math::sqrt(aMag);

        const int nodeR = meSCS->ipNodeMap(fdata.faceOrd)[ip];
        const int nodeL = meSCS->opposingNodes(fdata.faceOrd, ip);

        DoubleType ypBip = 0.0;
        for (int d = 0; d < FaceTraits::nDim_; ++d) {
          const DoubleType nj = v_area(ip, d) / aMag;
          const DoubleType ej = 0.25 * (v_coord(nodeR, d) - v_coord(nodeL, d));
          ypBip += nj * ej * nj * ej;
        }
        ypBip = stk::math::sqrt(ypBip);

        const int ni = faceIpNodeMap[ip];
        distOps(fdata, ni, 0) += aMag * ypBip;
        areaOps(fdata, ni, 0) += aMag;
      }
    });

  sierra::nalu::nalu_ngp::run_entity_algorithm(
    "unittest_basic_face_elem_nodal", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      wDist.get(mi, 0) /= wArea.get(mi, 0);
    });

  wArea.modify_on_device();
  wArea.sync_to_host();
  wDist.modify_on_device();
  wDist.sync_to_host();

  {
    double minArea = 1.0e20;
    double maxArea = -1.0e20;
    const double tol = 1.0e-15;
    const double wdistExpected = 0.25;
    const auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        const double* warea = stk::mesh::field_data(wallArea, node);
        const double* wdist = stk::mesh::field_data(wallNormDist, node);
        if (warea[0] < minArea)
          minArea = warea[0];
        if (warea[0] > maxArea)
          maxArea = warea[0];
        EXPECT_NEAR(wdist[0], wdistExpected, tol);
      }
    }
    EXPECT_NEAR(minArea, 0.25, tol);
    EXPECT_NEAR(maxArea, 1.0, tol);
  }
}

void
elem_loop_par_reduce(
  const stk::mesh::BulkData& bulk, sierra::nalu::ScalarFieldType& pressure)
{
  using Hex8Traits = sierra::nalu::AlgTraitsHex8;
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  const double presSet = 4.0;

  stk::mesh::field_fill(presSet, pressure);
  const auto& meta = bulk.mesh_meta_data();
  stk::mesh::Selector sel = meta.universal_part();
  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);

  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_dev(
      Hex8Traits::topo_);
  dataReq.add_cvfem_volume_me(meSCV);

  auto* coordsField = bulk.mesh_meta_data().coordinate_field();
  dataReq.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(pressure, 1);

  const unsigned presID = pressure.mesh_meta_data_ordinal();

  DoubleType pressureSum = 0.0;
  Kokkos::Sum<DoubleType> pressureReducer(pressureSum);

  sierra::nalu::nalu_ngp::run_elem_par_reduce(
    "unittest_elem_loop_par_reduce", meshInfo, stk::topology::ELEM_RANK,
    dataReq, sel,
    KOKKOS_LAMBDA(ElemSimdData & edata, DoubleType & pSum) {
      auto& scrViews = edata.simdScrView;
      auto& v_pres = scrViews.get_scratch_view_1D(presID);

      for (int i = 0; i < Hex8Traits::nodesPerElement_; ++i)
        pSum += v_pres(0);
    },
    pressureReducer);

  {
    const auto& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, sel);
    size_t numTotalNodes = 0;
    for (const auto* b : elemBuckets) {
      for (const auto elem : *b) {
        numTotalNodes += bulk.num_nodes(elem);
      }
    }

    double pSumCalc = 0.0;
    for (int i = 0; i < sierra::nalu::simdLen; ++i)
      pSumCalc += stk::simd::get_data(pressureSum, i);

    const double goldSum = presSet * numTotalNodes;
    const double tol = 1.0e-16;
    EXPECT_NEAR(goldSum, pSumCalc, tol);
  }
}

void
basic_face_elem_reduce(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordField,
  const sierra::nalu::GenericFieldType& exposedArea)
{
  using FaceTraits = sierra::nalu::AlgTraitsQuad4Hex8;
  using FaceSimdData =
    sierra::nalu::nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();

  sierra::nalu::ElemDataRequests faceData(meta);
  sierra::nalu::ElemDataRequests elemData(meta);
  auto meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      FaceTraits::FaceTraits::topo_);
  auto meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      FaceTraits::ElemTraits::topo_);

  faceData.add_cvfem_face_me(meFC);
  elemData.add_cvfem_surface_me(meSCS);
  faceData.add_coordinates_field(
    coordField, ndim, sierra::nalu::CURRENT_COORDINATES);
  faceData.add_face_field(
    exposedArea, FaceTraits::numFaceIp_, FaceTraits::nDim_);
  faceData.add_master_element_call(
    sierra::nalu::FC_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);
  elemData.add_coordinates_field(
    coordField, ndim, sierra::nalu::CURRENT_COORDINATES);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  stk::mesh::Part* part = meta.get_part("surface_5");
  stk::mesh::Selector sel(*part);

  const unsigned coordsID = coordField.mesh_meta_data_ordinal();
  const unsigned exposedAreaID = exposedArea.mesh_meta_data_ordinal();

  DoubleType totalWallDist = 0.0;
  Kokkos::Sum<DoubleType> distReducer(totalWallDist);
  sierra::nalu::nalu_ngp::run_face_elem_par_reduce(
    "unittest_basic_face_elem_reduce", meshInfo, faceData, elemData, sel,
    KOKKOS_LAMBDA(FaceSimdData & fdata, DoubleType & pSum) {
      auto& v_coord = fdata.simdElemView.get_scratch_view_2D(coordsID);
      auto& v_area = fdata.simdFaceView.get_scratch_view_2D(exposedAreaID);

      for (int ip = 0; ip < FaceTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d = 0; d < FaceTraits::nDim_; ++d)
          aMag += v_area(ip, d) * v_area(ip, d);
        aMag = stk::math::sqrt(aMag);

        const int nodeR = meSCS->ipNodeMap(fdata.faceOrd)[ip];
        const int nodeL = meSCS->opposingNodes(fdata.faceOrd, ip);

        DoubleType ypBip = 0.0;
        for (int d = 0; d < FaceTraits::nDim_; ++d) {
          const DoubleType nj = v_area(ip, d) / aMag;
          const DoubleType ej = 0.25 * (v_coord(nodeR, d) - v_coord(nodeL, d));
          ypBip += nj * ej * nj * ej;
        }
        ypBip = stk::math::sqrt(ypBip);

        pSum += ypBip;
      }
    },
    distReducer);

  double totWallDist = 0.0;
  const double totalWallDistExpected = 16.0;
  for (int i = 0; i < sierra::nalu::simdLen; ++i)
    totWallDist += stk::simd::get_data(totalWallDist, i);

  EXPECT_NEAR(totWallDist, totalWallDistExpected, 1.0e-15);
}

TEST_F(NgpLoopTest, NGP_basic_node_loop)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  basic_node_loop(*bulk, *pressure);
}

TEST_F(NgpLoopTest, NGP_basic_node_reduce)
{
  fill_mesh_and_init_fields("generated:16x16x16");

  basic_node_reduce(*bulk, *pressure);
}

TEST_F(NgpLoopTest, NGP_basic_node_reduce_array)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  basic_node_reduce_array(*bulk, *pressure, 3 * 3 * 3);
}

TEST_F(NgpLoopTest, NGP_basic_node_reduce_minmax)
{
  fill_mesh_and_init_fields("generated:16x16x16");

  basic_node_reduce_minmax(*bulk, 0.0, 16.0);
}

TEST_F(NgpLoopTest, NGP_basic_node_reduce_minmax_alt)
{
  fill_mesh_and_init_fields("generated:16x16x16");

  basic_node_reduce_minmax_alt(*bulk, 0.0, 16.0);
}

TEST_F(NgpLoopTest, NGP_basic_node_reduce_minmaxsum)
{
  fill_mesh_and_init_fields("generated:16x16x16");

  stk::mesh::Selector sel = bulk->mesh_meta_data().universal_part();
  const auto& bkts = bulk->get_buckets(stk::topology::NODE_RANK, sel);
  size_t numNodes = 0;
  for (auto* b : bkts) {
    numNodes += b->size();
  }

  basic_node_reduce_minmaxsum(*bulk, 0.0, 16.0, static_cast<double>(numNodes));
}

TEST_F(NgpLoopTest, NGP_basic_elem_loop)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  basic_elem_loop(*bulk, *pressure, *massFlowRate);
}

TEST_F(NgpLoopTest, NGP_basic_edge_loop)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  basic_edge_loop(*bulk, *pressure, *mdotEdge);
}

TEST_F(NgpLoopTest, NGP_elem_loop_scratch_views)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  elem_loop_scratch_views(*bulk, *pressure, *velocity);
}

TEST_F(NgpLoopTest, NGP_elem_loop_par_reduce)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  elem_loop_par_reduce(*bulk, *pressure);
}

TEST_F(NgpLoopTest, NGP_calc_mdot_elem_loop)
{
  fill_mesh_and_init_fields("generated:2x2x2");

  calc_mdot_elem_loop(*bulk, *density, *velocity, *massFlowRate);
}

TEST_F(NgpLoopTest, NGP_basic_face_elem_loop)
{
  if (bulk->parallel_size() > 1)
    return;

  auto& exposedAreaVec =
    meta->declare_field<double>(meta->side_rank(), "exposed_area_vector");
  auto& wallArea =
    meta->declare_field<double>(stk::topology::NODE_RANK, "wall_area");
  auto& wallNormDist =
    meta->declare_field<double>(stk::topology::NODE_RANK, "wall_normal_dist");

  stk::mesh::put_field_on_mesh(
    exposedAreaVec, meta->universal_part(),
    meta->spatial_dimension() * sierra::nalu::AlgTraitsQuad4::numScsIp_,
    nullptr);
  stk::mesh::put_field_on_mesh(wallArea, meta->universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(wallNormDist, meta->universal_part(), nullptr);

  fill_mesh_and_init_fields("generated:4x4x1|sideset:xXyYzZ");
  unit_test_kernel_utils::calc_exposed_area_vec(
    *bulk, sierra::nalu::AlgTraitsQuad4::topo_, *coordField, exposedAreaVec);

  basic_face_elem_loop(
    *bulk, *coordField, exposedAreaVec, wallArea, wallNormDist);
}

TEST_F(NgpLoopTest, NGP_basic_face_elem_reduce)
{
  if (bulk->parallel_size() > 1)
    return;

  auto& exposedAreaVec =
    meta->declare_field<double>(meta->side_rank(), "exposed_area_vector");
  stk::mesh::put_field_on_mesh(
    exposedAreaVec, meta->universal_part(),
    meta->spatial_dimension() * sierra::nalu::AlgTraitsQuad4::numScsIp_,
    nullptr);

  fill_mesh_and_init_fields("generated:4x4x1|sideset:xXyYzZ");
  unit_test_kernel_utils::calc_exposed_area_vec(
    *bulk, sierra::nalu::AlgTraitsQuad4::topo_, *coordField, exposedAreaVec);

  basic_face_elem_reduce(*bulk, *coordField, exposedAreaVec);
}
