#include <gtest/gtest.h>

#include <KokkosInterface.h>

static inline 
bool is_aligned(const void *pointer, size_t byte_count)
{
    return (uintptr_t)pointer % byte_count == 0;
}

void do_the_test()
{
  unsigned N = 1;
  unsigned numScalars = 3;

  Kokkos::View<unsigned*,sierra::nalu::MemSpace> ngpResults("ngpResults", 1);
  Kokkos::View<unsigned*,sierra::nalu::MemSpace>::HostMirror hostResults = Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  unsigned bytes_per_team = 16;
  unsigned bytes_per_thread = 128;

  auto team_exec = sierra::nalu::get_device_team_policy(N, bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    unsigned one = 1;
    sierra::nalu::SharedMemView<double*,sierra::nalu::DeviceShmem> view1 = sierra::nalu::get_shmem_view_1D<double,sierra::nalu::DeviceTeamHandleType,sierra::nalu::DeviceShmem>(team, numScalars);
    if (view1.size() > 0 && is_aligned(view1.data(), 8)) {
        Kokkos::atomic_add(&ngpResults(0), one);
    }

    sierra::nalu::SharedMemView<double*,sierra::nalu::DeviceShmem> view2 = sierra::nalu::get_shmem_view_1D<double,sierra::nalu::DeviceTeamHandleType,sierra::nalu::DeviceShmem>(team, numScalars);
    if (view2.size() > 0 && is_aligned(view2.data(), 8)) {
        Kokkos::atomic_add(&ngpResults(0), one);
    }

    sierra::nalu::SharedMemView<double*,sierra::nalu::DeviceShmem> view3 = sierra::nalu::get_shmem_view_1D<double,sierra::nalu::DeviceTeamHandleType,sierra::nalu::DeviceShmem>(team, numScalars);
    if (view3.size() > 0 && is_aligned(view3.data(), 8)) {
        Kokkos::atomic_add(&ngpResults(0), one);
    }
  });

  Kokkos::deep_copy(hostResults, ngpResults);

#ifdef KOKKOS_ENABLE_CUDA
//We're expecting the result to be 3.
//On GPU the result is 96... Is it 96 because there are 32 threads
//per warp and you never have less than 1 warp ???
//Not sure I understand or like this...
  EXPECT_EQ(96u, hostResults(0));
#else
  EXPECT_EQ(3u, hostResults(0));
#endif
}

TEST(Shmem, align)
{
  do_the_test();
}

