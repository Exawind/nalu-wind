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

  unsigned bytes_per_team = 0;
  unsigned bytes_per_thread = 2048;

  auto team_exec = sierra::nalu::get_device_team_policy(N, bytes_per_team, bytes_per_thread);
  unsigned numCorrectlyAlignedViews = 0;
  Kokkos::parallel_reduce(team_exec, KOKKOS_LAMBDA(const sierra::nalu::TeamHandleType& team, unsigned& localResult)
  {
    sierra::nalu::SharedMemView<double*> view1 = sierra::nalu::get_shmem_view_1D<double>(team, numScalars);
    if (is_aligned(view1.data(), 8)) {
        ++localResult;
    }

    sierra::nalu::SharedMemView<double*> view2 = sierra::nalu::get_shmem_view_1D<double>(team, numScalars);
    if (is_aligned(view2.data(), 8)) {
        ++localResult;
    }

    sierra::nalu::SharedMemView<double*> view3 = sierra::nalu::get_shmem_view_1D<double>(team, numScalars);
    if (is_aligned(view3.data(), 8)) {
        ++localResult;
    }
  }, numCorrectlyAlignedViews);

  EXPECT_EQ(3u, numCorrectlyAlignedViews);
}

TEST(Shmem, align)
{
}

