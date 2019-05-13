#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <SimdInterface.h>
#include <CopyAndInterleave.h>
#include <MultiDimViews.h>

using TeamType = sierra::nalu::DeviceTeamHandleType;
using ShmemType = sierra::nalu::DeviceShmem;

typedef Kokkos::DualView<int*, Kokkos::LayoutRight, sierra::nalu::DeviceSpace> IntViewType;

void do_the_interleave_test()
{
  const int bytes_per_team = 0;

  int numResults = 1;
  IntViewType result("result", numResults);

  int N = 4;
  int threads_per_team = 1;
  int bytes_per_thread = sizeof(double)*N*sierra::nalu::simdLen*2;
  auto team_exec = sierra::nalu::get_device_team_policy(1, bytes_per_team, bytes_per_thread, threads_per_team);

//  std::cout<<"simdLen = "<<sierra::nalu::simdLen<<std::endl;

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    sierra::nalu::SharedMemView<DoubleType*,ShmemType> simdView = sierra::nalu::get_shmem_view_1D<DoubleType,TeamType,ShmemType>(team, N);
    sierra::nalu::SharedMemView<double*,ShmemType> views[sierra::nalu::simdLen];
    const double* data[sierra::nalu::simdLen];
    for(int i=0; i<sierra::nalu::simdLen; ++i) {
      views[i] = sierra::nalu::get_shmem_view_1D<double,TeamType,ShmemType>(team, N);
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](const size_t&  /* index */)
    {
      for(int i=0; i<sierra::nalu::simdLen; ++i) {
        for(int j=0; j<N; ++j) {
          views[i](j) = j+1;
        }
        data[i] = views[i].data();
      }
  
      sierra::nalu::interleave(simdView, data, sierra::nalu::simdLen);
      result.d_view(0) = 1;
  
      for(int j=0; j<N; ++j) {
        for(int i=0; i<sierra::nalu::simdLen; ++i) {
          if (stk::simd::get_data(simdView(j), i) != j+1) {
            result.d_view(0) = 0;
          }
        }
      }
    });
  });

  result.modify<IntViewType::execution_space>();
  result.sync<IntViewType::host_mirror_space>();

  EXPECT_EQ(1, result.h_view(0));
}

TEST(CopyAndInterleave, interleave_1D)
{
  do_the_interleave_test();
}

template<typename ViewType>
bool check_view(const ViewType& v)
{
  unsigned len = v.size();
  typename ViewType::pointer_type ptr = v.data();
  for(unsigned i=0; i<len; ++i) {
    for(unsigned j=0; j<sierra::nalu::simdLen; ++j) {
      if (stk::simd::get_data(ptr[i], j) != j+1) {
        return false;
      }
    }
  }
  return true;
}

void do_the_multidimviews_test()
{
  int totalNumFields = 6;
  int numResults = totalNumFields;
  IntViewType result("result", numResults);

  sierra::nalu::NumNeededViews numNeededViews = {2, 2, 2, 0};

  int N = 4;
  const int bytes_per_team = 0;
  int threads_per_team = 1;
  int bytes_per_thread = sizeof(double)*2*(N + N*N + N*N*N)*sierra::nalu::simdLen*2
     + sierra::nalu::MultiDimViews<double>::bytes_needed(totalNumFields, numNeededViews)
     + sierra::nalu::simdLen * sierra::nalu::MultiDimViews<DoubleType>::bytes_needed(totalNumFields, numNeededViews);
  std::cout<<"bytes_per_thread = "<<bytes_per_thread<<std::endl;

  auto team_exec = sierra::nalu::get_device_team_policy(1, bytes_per_team, bytes_per_thread, threads_per_team);

  std::cout<<"simdLen = "<<sierra::nalu::simdLen<<std::endl;

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    unsigned maxOrdinal = totalNumFields-1;
    sierra::nalu::MultiDimViews<DoubleType> simdMultiDimViews(team, maxOrdinal, numNeededViews);

    simdMultiDimViews.add_1D_view(0, sierra::nalu::get_shmem_view_1D<DoubleType,TeamType,ShmemType>(team, N));
    simdMultiDimViews.add_1D_view(1, sierra::nalu::get_shmem_view_1D<DoubleType,TeamType,ShmemType>(team, N));
    simdMultiDimViews.add_2D_view(2, sierra::nalu::get_shmem_view_2D<DoubleType,TeamType,ShmemType>(team, N, N));
    simdMultiDimViews.add_2D_view(3, sierra::nalu::get_shmem_view_2D<DoubleType,TeamType,ShmemType>(team, N, N));
    simdMultiDimViews.add_3D_view(4, sierra::nalu::get_shmem_view_3D<DoubleType,TeamType,ShmemType>(team, N, N, N));
    simdMultiDimViews.add_3D_view(5, sierra::nalu::get_shmem_view_3D<DoubleType,TeamType,ShmemType>(team, N, N, N));

    std::unique_ptr<sierra::nalu::MultiDimViews<double>> multiDimViews[sierra::nalu::simdLen];
    for(int i=0; i<sierra::nalu::simdLen; ++i) {
      multiDimViews[i] = std::unique_ptr<sierra::nalu::MultiDimViews<double> >(new sierra::nalu::MultiDimViews<double>(team, maxOrdinal, numNeededViews));

      multiDimViews[i]->add_1D_view(0, sierra::nalu::get_shmem_view_1D<double,TeamType,ShmemType>(team, N));
      multiDimViews[i]->add_1D_view(1, sierra::nalu::get_shmem_view_1D<double,TeamType,ShmemType>(team, N));
      multiDimViews[i]->add_2D_view(2, sierra::nalu::get_shmem_view_2D<double,TeamType,ShmemType>(team, N, N));
      multiDimViews[i]->add_2D_view(3, sierra::nalu::get_shmem_view_2D<double,TeamType,ShmemType>(team, N, N));
      multiDimViews[i]->add_3D_view(4, sierra::nalu::get_shmem_view_3D<double,TeamType,ShmemType>(team, N, N, N));
      multiDimViews[i]->add_3D_view(5, sierra::nalu::get_shmem_view_3D<double,TeamType,ShmemType>(team, N, N, N));
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](const size_t&  /* index */)
    {
      for(int i=0; i<sierra::nalu::simdLen; ++i) {
        Kokkos::deep_copy(multiDimViews[i]->get_scratch_view_1D(0), i+1);
        Kokkos::deep_copy(multiDimViews[i]->get_scratch_view_1D(1), i+1);
        Kokkos::deep_copy(multiDimViews[i]->get_scratch_view_2D(2), i+1);
        Kokkos::deep_copy(multiDimViews[i]->get_scratch_view_2D(3), i+1);
        Kokkos::deep_copy(multiDimViews[i]->get_scratch_view_3D(4), i+1);
        Kokkos::deep_copy(multiDimViews[i]->get_scratch_view_3D(5), i+1);
      }
  
      const sierra::nalu::MultiDimViews<double>* multiDimViewPtrs[sierra::nalu::simdLen] = {nullptr};
      for(int i=0; i<sierra::nalu::simdLen; ++i) {
        multiDimViewPtrs[i] = multiDimViews[i].get();
      }
      sierra::nalu::copy_and_interleave(multiDimViewPtrs, sierra::nalu::simdLen, simdMultiDimViews);
  
      result.d_view(0) = check_view(simdMultiDimViews.get_scratch_view_1D(0)) ? 1 : 0;
      result.d_view(1) = check_view(simdMultiDimViews.get_scratch_view_1D(1)) ? 1 : 0;
      result.d_view(2) = check_view(simdMultiDimViews.get_scratch_view_2D(2)) ? 1 : 0;
      result.d_view(3) = check_view(simdMultiDimViews.get_scratch_view_2D(3)) ? 1 : 0;
      result.d_view(4) = check_view(simdMultiDimViews.get_scratch_view_3D(4)) ? 1 : 0;
      result.d_view(5) = check_view(simdMultiDimViews.get_scratch_view_3D(5)) ? 1 : 0;
    });
  });

  result.modify<IntViewType::execution_space>();
  result.sync<IntViewType::host_mirror_space>();

  for(int i=0; i<numResults; ++i) {
    EXPECT_EQ(1, result.h_view(i));
  }
}

TEST(CopyAndInterleave, multidimviews)
{
  do_the_multidimviews_test();
}

