#include "gtest/gtest.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

#include "UnitTestUtils.h"
#include "UnitTestRealm.h"

#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "ElemDataRequests.h"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Types.hpp"

void
test_ngp_mesh_1(
  const stk::mesh::BulkData& bulk, const stk::mesh::NgpMesh& ngpMesh)
{
  stk::topology elemTopo = stk::topology::HEX_8;

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local =
    meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets =
    bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  unsigned numStkBuckets = elemBuckets.size();
  unsigned numStkElements = 0;
  for (const stk::mesh::Bucket* b : elemBuckets) {
    numStkElements += b->size();
  }
  unsigned expectedNodesPerElem = elemTopo.num_nodes();

  Kokkos::View<unsigned*, sierra::nalu::MemSpace> ngpResults("ngpResults", 2);
  Kokkos::View<unsigned*, sierra::nalu::MemSpace>::HostMirror hostResults =
    Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(
    elemBuckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& b =
        ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());
      ++ngpResults(0);

      const size_t bucketLen = b.size();
      const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, simdBucketLen),
        [&](const size_t& bktIndex) {
          int numSimdElems =
            sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);

          for (int simdElemIndex = 0; simdElemIndex < numSimdElems;
               ++simdElemIndex) {
            stk::mesh::Entity element =
              b[bktIndex * sierra::nalu::simdLen + simdElemIndex];
            stk::mesh::FastMeshIndex elemIndex =
              ngpMesh.fast_mesh_index(element);
            if (
              ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex).size() ==
              expectedNodesPerElem) {
              unsigned one = 1;
              Kokkos::atomic_add(&ngpResults(1), one);
            }
          }
        });
    });

  Kokkos::deep_copy(hostResults, ngpResults);
  EXPECT_EQ(numStkBuckets, hostResults(0));
  EXPECT_EQ(numStkElements, hostResults(1));
}

TEST(NgpMesh, NGPMesh)
{
  const std::string meshSpec("generated:2x2x2");

  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());

  test_ngp_mesh_1(realm.bulk_data(), realm.ngp_mesh());
}

void
test_ngp_mesh_field_values(
  const stk::mesh::BulkData& bulk,
  VectorFieldType* velocity,
  GenericFieldType* massFlowRate)
{
  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local =
    meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets =
    bulk.get_buckets(stk::topology::ELEM_RANK, all_local);

  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpVelocity =
    stk::mesh::get_updated_ngp_field<double>(*velocity);
  stk::mesh::NgpField<double>& ngpMassFlowRate =
    stk::mesh::get_updated_ngp_field<double>(*massFlowRate);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(
    elemBuckets.size(), bytes_per_team, bytes_per_thread);

  const double xVel = 1.0;
  const double yVel = 2.0;
  const double zVel = 3.0;
  const double flowRate = 4.0;

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& b =
        ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

      const size_t bucketLen = b.size();
      const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, simdBucketLen),
        [&](const size_t& bktIndex) {
          int numSimdElems =
            sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);

          for (int simdElemIndex = 0; simdElemIndex < numSimdElems;
               ++simdElemIndex) {
            stk::mesh::Entity element =
              b[bktIndex * sierra::nalu::simdLen + simdElemIndex];
            stk::mesh::FastMeshIndex elemIndex =
              ngpMesh.fast_mesh_index(element);
            ngpMassFlowRate.get(elemIndex, 0) = flowRate;

            stk::mesh::NgpMesh::ConnectedNodes nodes =
              ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
            for (unsigned n = 0; n < nodes.size(); ++n) {
              ngpVelocity.get(ngpMesh, nodes[n], 0) = xVel;
              ngpVelocity.get(ngpMesh, nodes[n], 1) = yVel;
              ngpVelocity.get(ngpMesh, nodes[n], 2) = zVel;
            }
          }
        });
    });

  ngpVelocity.modify_on_device();
  ngpMassFlowRate.modify_on_device();
  ngpVelocity.sync_to_host();
  ngpMassFlowRate.sync_to_host();

  const double tol = 1.0e-16;
  for (const stk::mesh::Bucket* b : elemBuckets) {
    for (stk::mesh::Entity elem : *b) {
      const double* flowRateData = stk::mesh::field_data(*massFlowRate, elem);
      EXPECT_NEAR(flowRate, *flowRateData, tol);

      const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
      const unsigned numNodes = bulk.num_nodes(elem);
      for (unsigned n = 0; n < numNodes; ++n) {
        const double* velocityData = stk::mesh::field_data(*velocity, nodes[n]);
        EXPECT_NEAR(xVel, velocityData[0], tol);
        EXPECT_NEAR(yVel, velocityData[1], tol);
        EXPECT_NEAR(zVel, velocityData[2], tol);
      }
    }
  }
}

TEST_F(Hex8MeshWithNSOFields, NGPMeshField)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  test_ngp_mesh_field_values(*bulk, velocity, massFlowRate);
}

//struct MyBaseClass
//{
//  KOKKOS_DEFAULTED_FUNCTION MyBaseClass() = default;
//  KOKKOS_DEFAULTED_FUNCTION MyBaseClass(const MyBaseClass&) = default;
//  KOKKOS_DEFAULTED_FUNCTION virtual ~MyBaseClass() = default;
//
//  KOKKOS_FUNCTION virtual unsigned get_num() const { return 0; }
//};

struct MyDeviceClass //: public MyBaseClass
{
  KOKKOS_FUNCTION MyDeviceClass()
  : ngpField(), num(0)
  {
    printf("MyDeviceClass def ctor\n");
  }

  KOKKOS_FUNCTION MyDeviceClass(const MyDeviceClass& src)
  : ngpField(src.ngpField), num(src.num)
  {
    printf("MyDeviceClass copy ctor\n");
  }

  KOKKOS_FUNCTION ~MyDeviceClass()
  {
    printf("MyDeviceClass dtor\n");
  }

  KOKKOS_FUNCTION unsigned get_num() const /*override*/ { return num; }

  stk::mesh::NgpField<double> ngpField;
  unsigned num = 0;
};

void test_ngp_field_placement_new()
{
  MyDeviceClass hostObj;
  hostObj.num = 42;

  printf("sizeof(MyDeviceClass): %lu, sizeof(NgpField): %lu\n", sizeof(MyDeviceClass), sizeof(stk::mesh::NgpField<double>));
  std::string debugName("MyDeviceClass");
  MyDeviceClass* devicePtr = static_cast<MyDeviceClass*>(Kokkos::kokkos_malloc<stk::ngp::MemSpace>(debugName, 2*sizeof(MyDeviceClass)));

  int constructionFinished = 0;
  printf("about to call parallel_reduce for placement new\n");
  Kokkos::parallel_reduce(sierra::nalu::DeviceRangePolicy(0,1), KOKKOS_LAMBDA(const unsigned& i, int& localFinished) {
    printf("before placement-new\n");
    new (devicePtr) MyDeviceClass(hostObj);
    printf("after placement-new\n");
    localFinished = 1;
  }, constructionFinished);
  EXPECT_EQ(1, constructionFinished);

  int numFromDevice = 0;
  printf("about to call parallel_reduce for access check\n");
  Kokkos::parallel_reduce(sierra::nalu::DeviceRangePolicy(0,1), KOKKOS_LAMBDA(const unsigned& i, int& localNum) {
    localNum = devicePtr->get_num();
  }, numFromDevice);
  EXPECT_EQ(42, numFromDevice);
}

TEST(DevicePlacementNew, structWithNgpField)
{
  test_ngp_field_placement_new();
}

enum {MaxLen = 6};
typedef Kokkos::View<unsigned*, stk::ngp::MemSpace> UnsignedViewType;

class FakeFieldBase
{
public:
  KOKKOS_DEFAULTED_FUNCTION FakeFieldBase() = default;
  KOKKOS_DEFAULTED_FUNCTION FakeFieldBase(const FakeFieldBase&) = default;
  KOKKOS_DEFAULTED_FUNCTION FakeFieldBase(FakeFieldBase&&) = default;
  KOKKOS_FUNCTION FakeFieldBase& operator=(const FakeFieldBase&) { return *this; }
  KOKKOS_FUNCTION FakeFieldBase& operator=(FakeFieldBase&&) { return *this; }
  KOKKOS_FUNCTION virtual ~FakeFieldBase() {}
};

template<typename T>
struct FakeField : public FakeFieldBase
{
public:
  KOKKOS_FUNCTION FakeField()
  {
    printf("FakeField default ctor\n");
    for(int i=0; i<MaxLen; ++i) {
      val[i] = 1.1;
    }
  }
  KOKKOS_FUNCTION FakeField(const FakeField<T>& src)
  {
    printf("FakeField copy ctor\n");
    for(int i=0; i<MaxLen; ++i) {
      val[i] = src.val[i];
    }
  }
  KOKKOS_FUNCTION ~FakeField()
  {
    printf("FakeField dtor\n");
  }

  KOKKOS_FUNCTION double get_val() const /*override*/ { return val; }

private:
  UnsignedViewType unsignedDeviceView1;
  typename UnsignedViewType::HostMirror unsignedHostView1;
  UnsignedViewType unsignedDeviceView2;
  typename UnsignedViewType::HostMirror unsignedHostView2;
  UnsignedViewType unsignedDeviceView3;
  typename UnsignedViewType::HostMirror unsignedHostView3;
  UnsignedViewType unsignedDeviceView4;
  typename UnsignedViewType::HostMirror unsignedHostView4;
  UnsignedViewType unsignedDeviceView5;
  typename UnsignedViewType::HostMirror unsignedHostView5;
  UnsignedViewType unsignedDeviceView6;
  typename UnsignedViewType::HostMirror unsignedHostView6;
  UnsignedViewType unsignedDeviceView7;
  typename UnsignedViewType::HostMirror unsignedHostView7;
  UnsignedViewType unsignedDeviceView8;
  typename UnsignedViewType::HostMirror unsignedHostView8;
  UnsignedViewType unsignedDeviceView9;
  typename UnsignedViewType::HostMirror unsignedHostView9;
  T val[MaxLen];
};

struct MyFakeDeviceClass
{
  KOKKOS_DEFAULTED_FUNCTION MyFakeDeviceClass() = default;
  KOKKOS_FUNCTION MyFakeDeviceClass(const MyFakeDeviceClass& src)
  : ngpField(src.ngpField), num(src.num)
  {}
  KOKKOS_DEFAULTED_FUNCTION ~MyFakeDeviceClass() = default;

  KOKKOS_FUNCTION unsigned get_num() const /*override*/ { return num; }

  FakeField<double> ngpField;
  unsigned num = 0;
};

void test_fake_field_placement_new()
{
  MyFakeDeviceClass hostObj;
  hostObj.num = 42;

  printf("sizeof(MyFakeDeviceClass): %lu, sizeof(FakeField): %lu\n", sizeof(MyFakeDeviceClass), sizeof(FakeField<double>));
  std::string debugName("MyFakeDeviceClass");
  MyFakeDeviceClass* devicePtr = static_cast<MyFakeDeviceClass*>(Kokkos::kokkos_malloc<stk::ngp::MemSpace>(debugName, sizeof(MyFakeDeviceClass)));

  int constructionFinished = 0;
  printf("about to call parallel_reduce for placement new\n");
  Kokkos::parallel_reduce(sierra::nalu::DeviceRangePolicy(0,1), KOKKOS_LAMBDA(const unsigned& i, int& localFinished) {
    printf("before placement-new\n");
    new (devicePtr) MyFakeDeviceClass(hostObj);
    printf("after placement-new\n");
    localFinished = 1;
  }, constructionFinished);
  EXPECT_EQ(1, constructionFinished);

  int numFromDevice = 0;
  printf("about to call parallel_reduce for access check\n");
  Kokkos::parallel_reduce(sierra::nalu::DeviceRangePolicy(0,1), KOKKOS_LAMBDA(const unsigned& i, int& localNum) {
    localNum = devicePtr->get_num();
  }, numFromDevice);
  EXPECT_EQ(42, numFromDevice);
}

TEST(DevicePlacementNew, structWithFakeField)
{
  test_fake_field_placement_new();
}

