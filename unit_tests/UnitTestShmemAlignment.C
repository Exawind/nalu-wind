#include <gtest/gtest.h>

#include <KokkosInterface.h>

TEST(Shmem, align)
{
  unsigned N = 1;
  unsigned numScalars = 3;

  unsigned bytes_per_team = 0;
  unsigned bytes_per_thread = 2048;

  auto team_exec = sierra::nalu::get_device_team_policy(N, bytes_per_team, bytes_per_thread);
  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::TeamHandleType& team)
  {
    sierra::nalu::SharedMemView<double*> view1 = sierra::nalu::get_shmem_view_1D<double>(team, numScalars);
    std::cerr<<"view1.data(): "<<view1.data()<<", alignof(view1.data()): "<<alignof(view1.data())<<std::endl;

    sierra::nalu::SharedMemView<double*> view2 = sierra::nalu::get_shmem_view_1D<double>(team, numScalars);
    std::cerr<<"view2-view1: "<<ptrdiff_t(view2.data()-view1.data())*sizeof(double)<<", view2.data(): "<<view2.data()<<", alignof(view2.data()): "<<alignof(view2.data())<<std::endl;

    sierra::nalu::SharedMemView<double*> view3 = sierra::nalu::get_shmem_view_1D<double>(team, numScalars);
    std::cerr<<"view3-view2: "<<ptrdiff_t(view3.data()-view2.data())*sizeof(double)<<", view3.data(): "<<view3.data()<<", alignof(view3.data()): "<<alignof(view3.data())<<std::endl;
  });
}

