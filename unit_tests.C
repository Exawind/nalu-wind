// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h" // for InitGoogleTest, etc
#include "mpi.h"         // for MPI_Comm_rank, MPI_Finalize, etc
#include "Kokkos_Core.hpp"
#include "stk_util/parallel/Parallel.hpp"

#include "NaluVersionInfo.h"
#include "NaluEnv.h"
#include "master_element/MasterElementFactory.h"

int
main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  sierra::nalu::NaluEnv::self();
  Kokkos::initialize(argc, argv);
  int returnVal = 0;

#ifdef KOKKOS_ENABLE_CUDA
  const size_t nalu_stack_size = 16384;
  cudaDeviceSetLimit(cudaLimitStackSize, nalu_stack_size);
#endif
  // Create a dummy nested scope to ensure destructors are called before
  // Kokkos::finalize_all. The instances owning threaded Kokkos loops must be
  // cleared out before Kokkos::finalize is called.
  {
    // clang-format off
      namespace version = sierra::nalu::version;
      sierra::nalu::NaluEnv::self().naluOutputP0()
        << "   Nalu-Wind Version: " << version::NaluVersionTag << std::endl
        << "   Nalu-Wind GIT Commit SHA: " << version::NaluGitCommitSHA
        << ((version::RepoIsDirty == "DIRTY") ? ("-" + version::RepoIsDirty) : "") << std::endl
        << "   Trilinos Version: " << version::TrilinosVersionTag << std::endl << std::endl;
    // clang-format on

    testing::InitGoogleTest(&argc, argv);

    returnVal = RUN_ALL_TESTS();

    // Force deallocation of all MasterElements created. This is necessary
    // when specific unit tests are run using the gtest_filter option that
    // provides no mechanism for call the destructors of the master elements
    // created for those tests.
    sierra::nalu::MasterElementRepo::clear();
  }

  Kokkos::finalize_all();
  MPI_Finalize();

  return returnVal;
}
