#include <gtest/gtest.h>

#include <KokkosInterface.h>

KOKKOS_FUNCTION inline bool
is_aligned(const void* pointer, size_t byte_count)
{
  return (uintptr_t)pointer % byte_count == 0;
}

void
do_the_test()
{
  unsigned N = 1;
  unsigned numScalars = 3;

  Kokkos::View<unsigned*, sierra::kynema-ugf::MemSpace> ngpResults("ngpResults", 1);
  Kokkos::View<unsigned*, sierra::kynema-ugf::MemSpace>::HostMirror hostResults =
    Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  unsigned bytes_per_team = 16;
  unsigned bytes_per_thread = 128;
  unsigned threads_per_team = 1;

  auto team_exec = sierra::kynema-ugf::get_device_team_policy(
    N, bytes_per_team, bytes_per_thread, threads_per_team);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::kynema-ugf::DeviceTeamHandleType& team) {
      unsigned one = 1;
      sierra::kynema-ugf::SharedMemView<double*, sierra::kynema-ugf::DeviceShmem> view1 =
        sierra::kynema-ugf::get_shmem_view_1D<
          double, sierra::kynema-ugf::DeviceTeamHandleType,
          sierra::kynema-ugf::DeviceShmem>(team, numScalars);
      if (view1.size() > 0 && is_aligned(view1.data(), 8)) {
        Kokkos::atomic_add(&ngpResults(0), one);
      }

      sierra::kynema-ugf::SharedMemView<double*, sierra::kynema-ugf::DeviceShmem> view2 =
        sierra::kynema-ugf::get_shmem_view_1D<
          double, sierra::kynema-ugf::DeviceTeamHandleType,
          sierra::kynema-ugf::DeviceShmem>(team, numScalars);
      if (view2.size() > 0 && is_aligned(view2.data(), 8)) {
        Kokkos::atomic_add(&ngpResults(0), one);
      }

      sierra::kynema-ugf::SharedMemView<double*, sierra::kynema-ugf::DeviceShmem> view3 =
        sierra::kynema-ugf::get_shmem_view_1D<
          double, sierra::kynema-ugf::DeviceTeamHandleType,
          sierra::kynema-ugf::DeviceShmem>(team, numScalars);
      if (view3.size() > 0 && is_aligned(view3.data(), 8)) {
        Kokkos::atomic_add(&ngpResults(0), one);
      }
    });

  Kokkos::deep_copy(hostResults, ngpResults);

  EXPECT_EQ(3u, hostResults(0));
}

TEST(Shmem, align) { do_the_test(); }
