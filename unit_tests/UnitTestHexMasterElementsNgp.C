#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <array>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

#include <master_element/Hex8CVFEM.h>
#include <master_element/Hex27CVFEM.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"
#include "Kokkos_Array.hpp"
#include <stk_ngp/Ngp.hpp>

#include "UnitTestUtils.h"
#include "utils/CreateDeviceExpression.h"


using TeamType = sierra::nalu::DeviceTeamHandleType;
using ShmemType = sierra::nalu::DeviceShmem;

namespace {

  // evaluate a polynomial, left-to-right
template <int ORD>
KOKKOS_FUNCTION
double poly_val(const Kokkos::Array<double,ORD+1>& coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j] * std::pow(x,j);
  }
  return val;
}

template <int DIM, int ORD>
KOKKOS_FUNCTION
double poly_val(
  const Kokkos::Array<Kokkos::Array<double,ORD+1>,DIM>& coeffs, const double* x)
{
  if( coeffs.size() == 2) {
    return (poly_val<ORD>(coeffs[0],x[0]) * poly_val<ORD>(coeffs[1],x[1]));
  }
  return (poly_val<ORD>(coeffs[0],x[0]) * poly_val<ORD>(coeffs[1],x[1]) * poly_val<ORD>(coeffs[2],x[2]));
}

template <int ORD>
KOKKOS_FUNCTION
double poly_der(const Kokkos::Array<double,ORD+1>& coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += j*coeffs[j] * std::pow(x,j-1);
  }
  return val;
}

template <int DIM, int ORD>
KOKKOS_FUNCTION
double poly_der(
  const Kokkos::Array<Kokkos::Array<double,ORD+1>,DIM>& coeffs, const double* x, int dir)
{
  double val = 0.0;


  if( coeffs.size() == 2) {
    switch(dir) {
      case 0:
        val = poly_der<ORD>(coeffs[0], x[0]) * poly_val<ORD>(coeffs[1], x[1]);
        break;
      case 1:
        val = poly_val<ORD>(coeffs[0], x[0]) * poly_der<ORD>(coeffs[1], x[1]);
        break;
      default:
        printf("invalid direction");
        NGP_ThrowAssert(true);
    }
  }

  if( coeffs.size() == 3) {
    switch(dir) {
      case 0:
          val = poly_der<ORD>(coeffs[0], x[0]) * poly_val<ORD>(coeffs[1], x[1]) * poly_val<ORD>(coeffs[2], x[2]);
          break;
        case 1:
          val = poly_val<ORD>(coeffs[0], x[0]) * poly_der<ORD>(coeffs[1], x[1]) * poly_val<ORD>(coeffs[2], x[2]);
          break;
        case 2:
          val = poly_val<ORD>(coeffs[0], x[0]) * poly_val<ORD>(coeffs[1], x[1]) * poly_der<ORD>(coeffs[2], x[2]);
          break;
      default:
        printf("invalid direction");
        NGP_ThrowAssert(true);
    }
  }

  return val;
}

template <typename AlgTraits, typename ME, bool SCS>
void check_interpolation(
  const stk::mesh::MetaData& meta,
  const stk::mesh::BulkData& bulk)
{
  // Check that we can interpolate a random 3D polynomial of order poly_order
  // to the integration points

  const int dim        = AlgTraits::nDim_;
  const int num_nodes  = AlgTraits::nodesPerElement_;
  const int num_int_pt = SCS ? AlgTraits::numScsIp_ : AlgTraits::numScvIp_;
  const int poly_order = num_nodes == 8 ? 1 : 2;
 
  ngp::Mesh ngpMesh(bulk);

  ME    *me = SCS ? 
    dynamic_cast<ME*>(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)):
    dynamic_cast<ME*>(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_));
  ME *ngpMe = SCS ? 
    dynamic_cast<ME*>(sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>()):
    dynamic_cast<ME*>(sierra::nalu::MasterElementRepo::get_volume_master_element<AlgTraits>());
  ThrowRequire(me);
  ThrowRequire(ngpMe);

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);
  EXPECT_EQ(elems.size(), 1u); // single element test

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  // get random polynomial
  Kokkos::Array<Kokkos::Array<double,poly_order+1>,dim> coeffs;
  for (int j = 0; j < dim; ++j) {
    for (int i = 0; i < poly_order+1; ++i) {
      coeffs[j][i] = coeff(rng);
    }
  }

  std::vector<double> polyResult(num_int_pt);
  for (int j = 0; j < num_int_pt; ++j) {
    polyResult[j] = poly_val<dim,poly_order>(coeffs, &me->intgLoc_[j*dim]);
  }

  const auto* const coordField = bulk.mesh_meta_data().coordinate_field();
  EXPECT_TRUE(coordField != nullptr);
  ngp::Field<double> ngpCoordField(bulk, *coordField);
  ngpCoordField.copy_host_to_device(bulk, *coordField);

  Kokkos::View<DoubleType*,sierra::nalu::MemSpace> ngpResults("ngpResults", num_int_pt);
  Kokkos::View<DoubleType*,sierra::nalu::MemSpace>::HostMirror hostResults = Kokkos::create_mirror_view(ngpResults);
  for (int j = 0 ; j < num_int_pt; ++j) hostResults(j) = 0;
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::SharedMemView<DoubleType**>::shmem_size(num_int_pt, num_nodes);

  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();

  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  auto team_exec = sierra::nalu::get_device_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

    using ViewType = sierra::nalu::SharedMemView<DoubleType**,ShmemType>;
    ViewType shpfc = sierra::nalu::get_shmem_view_2D<DoubleType,TeamType,ShmemType>(team, num_int_pt, num_nodes);

    const size_t bucketLen   = b.size();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex)
    {
      const stk::mesh::Entity element = b[bktIndex];
      const stk::mesh::FastMeshIndex elemIndex = ngpMesh.fast_mesh_index(element);

      const ngp::Mesh::ConnectedNodes nodes = ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
      Kokkos::Array<double,num_nodes> ws_field;
      for (int n = 0; n < num_nodes; ++n) {
        Kokkos::Array<double,3> coords;
        for (int i = 0; i < 3; ++i) 
          coords[i] = ngpCoordField.get(ngpMesh, nodes[n], i);
        ws_field[n] = poly_val<dim,poly_order>(coeffs, coords.data());
      }
      ngpMe->template shape_fcn<ViewType>(shpfc);
      for (int j = 0; j < num_int_pt; ++j) {
        for (int i = 0; i < num_nodes; ++i) {
          ngpResults[j] += shpfc(j,i) * ws_field[i];
        }
      }
    });
  });
  
  Kokkos::deep_copy(hostResults, ngpResults);
  for (int j = 0 ; j < num_int_pt; ++j) {
    EXPECT_NEAR(stk::simd::get_data(hostResults(j),0), polyResult[j], 1.0e-12);
  }
}

template <typename AlgTraits>
void check_derivatives(
  const stk::mesh::MetaData& meta,
  const stk::mesh::BulkData& bulk)
{
  // Check that we can interpolate a random 3D polynomial
  // to the integration points

  using ME             = typename AlgTraits::masterElementScs_;
  const int dim        = AlgTraits::nDim_;
  const int num_nodes  = AlgTraits::nodesPerElement_;
  const int num_int_pt = AlgTraits::numScsIp_;
  const int poly_order = num_nodes == 8 ? 1 : 2;
 
  ngp::Mesh ngpMesh(bulk);

  ME    *me = dynamic_cast<ME*>(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_));
  ME *ngpMe = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();
  ThrowRequire(me);
  ThrowRequire(ngpMe);

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);
  EXPECT_EQ(elems.size(), 1u); // single element test

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  // get random polynomial
  Kokkos::Array<Kokkos::Array<double,poly_order+1>,dim> coeffs;
  for (unsigned j = 0; j < dim; ++j) {
    for (unsigned i = 0; i < coeffs[j].size(); ++i) {
      coeffs[j][i] = coeff(rng);
    }
  }

  std::array<std::array<double,dim>,num_int_pt> polyResult;
  for (int j = 0; j < num_int_pt; ++j) {
    for (unsigned d = 0; d < dim; ++d) {
      polyResult[j][d] = poly_der<dim,poly_order>(coeffs, &me->intgLoc_[j*dim], d);
    }
  }

  const auto* const coordField = bulk.mesh_meta_data().coordinate_field();
  EXPECT_TRUE(coordField != nullptr);
  ngp::Field<double> ngpCoordField(bulk, *coordField);
  ngpCoordField.copy_host_to_device(bulk, *coordField);

  Kokkos::View<DoubleType**,sierra::nalu::MemSpace> ngpResults("ngpResults", num_int_pt, dim);
  Kokkos::View<DoubleType**,sierra::nalu::MemSpace>::HostMirror hostResults = Kokkos::create_mirror_view(ngpResults);
  for (int j = 0 ; j < num_int_pt; ++j) 
    for (int d = 0 ; d < dim; ++d) 
      hostResults(j,d) = 0;
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 
      sierra::nalu::SharedMemView<DoubleType** >::shmem_size(            num_nodes, dim) +
    2*sierra::nalu::SharedMemView<DoubleType***>::shmem_size(num_int_pt, num_nodes, dim);

  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();

  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  auto team_exec = sierra::nalu::get_device_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

    sierra::nalu::SharedMemView<DoubleType**,ShmemType> coords = 
       sierra::nalu::get_shmem_view_2D<DoubleType,TeamType,ShmemType>(team,             num_nodes, dim);
    sierra::nalu::SharedMemView<DoubleType***,ShmemType> meGrad = 
       sierra::nalu::get_shmem_view_3D<DoubleType,TeamType,ShmemType>(team, num_int_pt, num_nodes, dim);
    sierra::nalu::SharedMemView<DoubleType***,ShmemType> meDeriv = 
       sierra::nalu::get_shmem_view_3D<DoubleType,TeamType,ShmemType>(team, num_int_pt, num_nodes, dim);

    const size_t bucketLen   = b.size();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex)
    {
      const stk::mesh::Entity element = b[bktIndex];
      const stk::mesh::FastMeshIndex elemIndex = ngpMesh.fast_mesh_index(element);

      const ngp::Mesh::ConnectedNodes nodes = ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
      Kokkos::Array<double,num_nodes> ws_field;
      for (int n = 0; n < num_nodes; ++n) {
        Kokkos::Array<double,3> cord;
        for (int i = 0; i < 3; ++i) {
          cord[i] = ngpCoordField.get(ngpMesh, nodes[n], i);
          coords(n,i) = cord[i];
        }
        ws_field[n] = poly_val<dim,poly_order>(coeffs, cord.data());
      }
      ngpMe->grad_op(coords, meGrad, meDeriv);
      for (int j = 0; j < num_int_pt; ++j) {
        for (int i = 0; i < num_nodes; ++i) {
          for (unsigned d = 0; d < dim; ++d) {
            ngpResults(j,d) += meGrad(j,i,d) * ws_field[i];
          }
        }
      }
    });
  });
  
  Kokkos::deep_copy(hostResults, ngpResults);
  for (int j = 0; j < num_int_pt; ++j) {
    for (unsigned d = 0; d < dim; ++d) {
      EXPECT_NEAR(stk::simd::get_data(hostResults(j,d),0), polyResult[j][d], 1.0e-12);
    }
  }
  sierra::nalu::MasterElementRepo::clear();
}

class MasterElementHexSerialNGP : public ::testing::Test
{
protected:
    MasterElementHexSerialNGP()
    : comm(MPI_COMM_WORLD), spatialDimension(3),
      meta(spatialDimension), bulk(meta, comm),
      ngpMesh(bulk),
      poly_order(1), topo(stk::topology::HEX_8)
    {
    }

    void setup_poly_order_1_hex_8() {
      poly_order = 1;
      topo = stk::topology::HEX_8;
      unit_test_utils::create_one_reference_element(bulk, stk::topology::HEX_8);
    }

    void setup_poly_order_2_hex_27() {
      poly_order = 2;
      topo = stk::topology::HEX_27;
      unit_test_utils::create_one_reference_element(bulk, stk::topology::HEX_27);
    }

    stk::ParallelMachine comm;
    unsigned spatialDimension;
    stk::mesh::MetaData meta;
    stk::mesh::BulkData bulk;
    ngp::Mesh ngpMesh;
    unsigned poly_order;
    stk::topology topo;
};


TEST_F(MasterElementHexSerialNGP, hex8_scs_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_1_hex_8();
    using AlgTraits = sierra::nalu::AlgTraitsHex8;
    check_interpolation<AlgTraits, AlgTraits::masterElementScs_, true> (meta, bulk);
  }
}

TEST_F(MasterElementHexSerialNGP, hex8_scv_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_1_hex_8();
    using AlgTraits = sierra::nalu::AlgTraitsHex8;
    check_interpolation<AlgTraits, AlgTraits::masterElementScv_, false>(meta, bulk);
  }
}

TEST_F(MasterElementHexSerialNGP, hex8_scs_derivatives)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_1_hex_8();
    check_derivatives<sierra::nalu::AlgTraitsHex8>(meta, bulk);
  }
}

TEST_F(MasterElementHexSerialNGP, hex27_scs_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_2_hex_27();
    //using AlgTraits = sierra::nalu::AlgTraitsHex27;
    //check_interpolation<AlgTraits, AlgTraits::masterElementScs_, true>(meta, bulk);
  }
}

TEST_F(MasterElementHexSerialNGP, hex27_scv_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_2_hex_27();
    sierra::nalu::Hex27SCV hex27scv;
    //check_interpolation<sierra::nalu::AlgTraitsHex27>(meta, bulk, hex27scv);
  }
}

TEST_F(MasterElementHexSerialNGP, hex27_scs_derivatives)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_2_hex_27();
    sierra::nalu::Hex27SCS hex27scs;
    //check_derivatives<3,2>(bulk, topo, hex27scs);
  }
}

}//namespace

