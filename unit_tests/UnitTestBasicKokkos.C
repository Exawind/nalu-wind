#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <Kokkos_Core.hpp>

TEST(BasicKokkos, discover_execution_space)
{
    stk::ParallelMachine comm = MPI_COMM_WORLD;
    int proc = stk::parallel_machine_rank(comm);

    if (proc == 0) {
        std::cout << std::endl;

#ifdef KOKKOS_HAVE_SERIAL
        std::cout << "Kokkos::Serial is available." << std::endl;
#endif

#ifdef KOKKOS_HAVE_OPENMP
        std::cout << "Kokkos::OpenMP is available. (Control num-threads via env-var OMP_NUM_THREADS)" << std::endl;
#endif

#ifdef KOKKOS_ENABLE_CUDA
        std::cout << "Kokkos::Cuda is available." << std::endl;
#endif
        std::cout << "Default execution space info: ";
        Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

        std::cout << std::endl;
    }
}

TEST(BasicKokkos, simple_views_1D)
{
    const double tolerance = 0.0000001;
    const size_t N = 10;
    Kokkos::View<double*> device_view1D("device_view1D", N);
    Kokkos::View<double*>::HostMirror host_view1D = Kokkos::create_mirror_view(device_view1D);
    for(size_t i=0; i<N; ++i) {
        host_view1D(i) = i+1;
    }

    Kokkos::deep_copy(device_view1D, host_view1D);

    Kokkos::View<double*>::HostMirror host_view1D_2("host_view1D_2", N);
    Kokkos::deep_copy(host_view1D_2, device_view1D);

    for(size_t i=0; i<N; ++i) {
        EXPECT_NEAR(host_view1D(i), host_view1D_2(i), tolerance);
    }
}

TEST(BasicKokkos, simple_views_2D)
{
    const double tolerance = 0.0000001;
    const size_t N = 10;
    const size_t M = 20;
    Kokkos::View<double**> device_view2D("device_view2D", N, M);
    Kokkos::View<double**>::HostMirror host_view2D = Kokkos::create_mirror_view(device_view2D);
    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<M; ++j) {
            host_view2D(i,j) = i+j+1;
        }
    }

    Kokkos::deep_copy(device_view2D, host_view2D);

    Kokkos::View<double**>::HostMirror host_view2D_2("host_view2D_2", N, M);
    Kokkos::deep_copy(host_view2D_2, device_view2D);

    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<M; ++j) {
            EXPECT_NEAR(host_view2D(i,j), host_view2D_2(i,j), tolerance);
        }
    }
}

void run_parallel_for_test()
{
    const double tolerance = 0.0000001;
    const size_t N = 10;
    const size_t M = 20;
    Kokkos::View<double**> device_view2D("host_view2D", N, M);
    Kokkos::View<double**>::HostMirror host_view2D = Kokkos::create_mirror_view(device_view2D);

    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<M; ++j) {
            host_view2D(i,j) = i+j+1;
        }
    }

    Kokkos::deep_copy(device_view2D, host_view2D);

//Important note: when the 'host' and 'device' share the same memory space, (as is the case for OpenMP),
//device_vew2D is semantically just a pointer to host_view2D, and the deep_copy is a no-op.
//That means that the parallel_for which comes next, is updating the values of host_view2D.
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const size_t& i) {
        for(size_t j=0; j<M; ++j) {
            device_view2D(i, j) *= 2;
        }
    });

//This deep_copy is a no-op for OpenMP, but for Cuda it is necessary; otherwise the values
//in host_view2D would not be updated and the following EXPECT_NEAR checks would fail.
    Kokkos::deep_copy(host_view2D, device_view2D);

    Kokkos::View<double**>::HostMirror host_result("host_result", N, M);
    Kokkos::deep_copy(host_result, device_view2D);

    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<M; ++j) {
            EXPECT_NEAR(host_result(i,j), host_view2D(i,j), tolerance);
        }
    }
}

TEST(BasicKokkos, parallel_for)
{
    run_parallel_for_test();
}

void run_nested_parallel_for_thread_teams_test()
{
    const double tolerance = 0.0000001;
    const size_t N = 8;
    const size_t M = 8;
    Kokkos::View<double**> device_view2D("device_view2D", N, M);
    Kokkos::View<double**>::HostMirror host_view2D = Kokkos::create_mirror_view(device_view2D);

    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<M; ++j) {
            host_view2D(i,j) = i+j+1;
        }
    }

    Kokkos::deep_copy(device_view2D, host_view2D);

    typedef Kokkos::Schedule<Kokkos::Dynamic> DynamicScheduleType;
    typedef typename Kokkos::TeamPolicy<typename Kokkos::DefaultExecutionSpace, DynamicScheduleType>::member_type TeamHandleType;

//Important note: when the 'host' and 'device' share the same memory space, (as is the case for OpenMP),
//device_vew2D is semantically just a pointer to host_view2D, and the deep_copy is a no-op.
//That means that the parallel_for which comes next, is updating the values of host_view2D.
    Kokkos::parallel_for(Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamHandleType& team) {
            size_t i = team.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, (size_t)0, M), [&](const size_t& j) {
                device_view2D(i, j) *= 2;
            });
        });

//This deep_copy is a no-op for OpenMP, but for Cuda it is necessary; otherwise the values
//in host_view2D would not be updated and the following EXPECT_NEAR checks would fail.
    Kokkos::deep_copy(host_view2D, device_view2D);

    Kokkos::View<double**>::HostMirror host_result("host_result", N, M);
    Kokkos::deep_copy(host_result, device_view2D);

    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<M; ++j) {
            EXPECT_NEAR(host_result(i,j), host_view2D(i,j), tolerance);
        }
    }
}

TEST(BasicKokkos, nested_parallel_for_thread_teams)
{
    run_nested_parallel_for_thread_teams_test();
}

