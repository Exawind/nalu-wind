#include "gtest/gtest.h"

#include "KokkosInterface.h"

void
test_nalu_kokkos_pll_for()
{
  Kokkos::View<unsigned*, sierra::nalu::MemSpace> ngpResults("ngpResults", 2);
  Kokkos::View<unsigned*, sierra::nalu::MemSpace>::HostMirror hostResults =
    Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(1, bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      constexpr unsigned one = 1;
      Kokkos::atomic_add(&ngpResults(0), one);

      const int innerLoopLength = 1;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, innerLoopLength),
        [&](const size_t& idx) {
          Kokkos::atomic_add(&ngpResults(1), one);
        });
    });

  Kokkos::deep_copy(hostResults, ngpResults);
  std::cout << "outer-loop result: "<<hostResults(0)<<std::endl;
  std::cout << "inner-loop result: "<<hostResults(1)<<std::endl;
}

TEST(Kokkos, nestedParallelFor)
{
  test_nalu_kokkos_pll_for();
}

void
test_pure_kokkos_auto_pll_for()
{
  using myDeviceSpace = Kokkos::DefaultExecutionSpace;
  using myMemSpace = myDeviceSpace::memory_space;
  using myDeviceTeamPolicy = Kokkos::TeamPolicy<myDeviceSpace>;
  using myDynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
  using myDeviceTeamHandleType = Kokkos::TeamPolicy<myDeviceSpace, myDynamicScheduleType>::member_type;

  Kokkos::View<unsigned*, myMemSpace> ngpResults("ngpResults", 2);
  Kokkos::View<unsigned*, myMemSpace>::HostMirror hostResults =
    Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  const int loopLength = 1;
  myDeviceTeamPolicy policy(loopLength, Kokkos::AUTO);
  auto team_exec = policy.set_scratch_size(loopLength, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const myDeviceTeamHandleType& team) {
      constexpr unsigned one = 1;
      Kokkos::atomic_add(&ngpResults(0), one);

      const int innerLoopLength = 1;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, innerLoopLength),
        [&](const size_t& idx) {
          Kokkos::atomic_add(&ngpResults(1), one);
        });
    });

  Kokkos::deep_copy(hostResults, ngpResults);
  std::cout << "outer-loop result: "<<hostResults(0)<<std::endl;
  std::cout << "inner-loop result: "<<hostResults(1)<<std::endl;
}

TEST(Kokkos, pureKokkos_AUTO_nestedParallelFor)
{
  test_pure_kokkos_auto_pll_for();
}

void
test_pure_kokkos_no_auto_pll_for()
{
  using myDeviceSpace = Kokkos::DefaultExecutionSpace;
  using myMemSpace = myDeviceSpace::memory_space;
  using myDeviceTeamPolicy = Kokkos::TeamPolicy<myDeviceSpace>;
  using myDynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
  using myDeviceTeamHandleType = Kokkos::TeamPolicy<myDeviceSpace, myDynamicScheduleType>::member_type;

  Kokkos::View<unsigned*, myMemSpace> ngpResults("ngpResults", 2);
  Kokkos::View<unsigned*, myMemSpace>::HostMirror hostResults =
    Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  const int loopLength = 1;
  const int threadsPerTeam = 64;
  myDeviceTeamPolicy policy(loopLength, threadsPerTeam);
  auto team_exec = policy.set_scratch_size(loopLength, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const myDeviceTeamHandleType& team) {
      constexpr unsigned one = 1;
      Kokkos::atomic_add(&ngpResults(0), one);

      const int innerLoopLength = 1;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, innerLoopLength),
        [&](const size_t& idx) {
          Kokkos::atomic_add(&ngpResults(1), one);
        });
    });

  Kokkos::deep_copy(hostResults, ngpResults);
  std::cout << "outer-loop result: "<<hostResults(0)<<std::endl;
  std::cout << "inner-loop result: "<<hostResults(1)<<std::endl;
}

TEST(Kokkos, pureKokkos_NoAuto_nestedParallelFor)
{
  test_pure_kokkos_no_auto_pll_for();
}

