// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Simulation.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <KynemaUGFParsing.h>
#include <KynemaUGFEnv.h>
#include <Realms.h>
#include <xfer/Transfers.h>
#include <TimeIntegrator.h>
#include <LinearSolvers.h>
#include <KynemaUGFVersionInfo.h>
#include "overset/ExtOverset.h"

#include <Ioss_SerializeIO.h>

namespace sierra {
namespace kynema_ugf {

//==========================================================================
// Class Definition
//==========================================================================
// Simulation - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------

Simulation::Simulation(const YAML::Node& root_node)
  : m_root_node(root_node),
    timeIntegrator_(NULL),
    realms_(NULL),
    transfers_(NULL),
    linearSolvers_(NULL),
    serializedIOGroupSize_(0)
{
#if defined(KOKKOS_ENABLE_CUDA)
  cudaDeviceGetLimit(&default_stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, kynema_ugf_stack_size);
#elif defined(KOKKOS_ENABLE_HIP)
  hipError_t err = hipDeviceGetLimit(&default_stack_size, hipLimitStackSize);
  if (err != hipSuccess) {
    /*
     This might be useful at some point so keeping it and commenting out.

     printf("%s %s %d : Failure %s in hipDeviceSetLimit\n",
     __FILE__,__FUNCTION__,__LINE__,hipGetErrorString(err));
    */
  }

  err = hipDeviceSetLimit(hipLimitStackSize, kynema_ugf_stack_size);
  if (err != hipSuccess) {
    /*
     This might be useful at some point so keeping it and commenting out.

     printf("%s %s %d : Failure %s in hipDeviceSetLimit\n",
     __FILE__,__FUNCTION__,__LINE__,hipGetErrorString(err));
    */
  }
#endif
}

Simulation::~Simulation()
{
  delete realms_;
  delete transfers_;
  delete timeIntegrator_;
  delete linearSolvers_;
#if defined(KOKKOS_ENABLE_CUDA)
  cudaDeviceSetLimit(cudaLimitStackSize, default_stack_size);
#elif defined(KOKKOS_ENABLE_HIP)
  hipError_t err = hipDeviceSetLimit(hipLimitStackSize, default_stack_size);
  if (err != hipSuccess) {
    /*
     This might be useful at some point so keeping it and commenting out.

     printf("%s %s %d : Failure %s in hipDeviceSetLimit\n",
     __FILE__,__FUNCTION__,__LINE__,hipGetErrorString(err));
    */
  }
#endif
}

// Timers
// static
stk::diag::TimerSet&
Simulation::rootTimerSet()
{
  static stk::diag::TimerSet s_timerSet(sierra::Diag::TIMER_ALL);

  return s_timerSet;
}

// static
stk::diag::Timer&
Simulation::rootTimer()
{
  static stk::diag::Timer s_timer =
    stk::diag::createRootTimer("KynemaUGF", rootTimerSet());

  return s_timer;
}

// static
stk::diag::Timer&
Simulation::outputTimer()
{
  static stk::diag::Timer s_timer("Output", rootTimer());
  return s_timer;
}

void
Simulation::load(const YAML::Node& node)
{

  high_level_banner();

  // load the linear solver configs
  linearSolvers_ = new LinearSolvers(*this);
  linearSolvers_->load(node);

  // create the realms
  realms_ = new Realms(*this);
  realms_->load(node);

  // create the time integrator
  KynemaUGFEnv::self().kynema_ugfOutputP0() << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "Time Integrator Review:  " << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "=========================" << std::endl;
  timeIntegrator_ = new TimeIntegrator(this);
  timeIntegrator_->load(node);

  // create the transfers; mesh is already loaded in realm
  KynemaUGFEnv::self().kynema_ugfOutputP0() << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "Transfer Review:         " << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "=========================" << std::endl;
  transfers_ = new Transfers(*this);
  transfers_->load(node);
}

void
Simulation::setSerializedIOGroupSize(int siogs)
{
  if (siogs) {
    if (
      siogs < 0 || siogs > KynemaUGFEnv::self().parallel_size() ||
      KynemaUGFEnv::self().parallel_size() % siogs != 0) {
      KynemaUGFEnv::self().kynema_ugfOutputP0()
        << "Error: Job requested serialized_io_group_size of " << siogs
        << " which is incompatible with MPI size= "
        << KynemaUGFEnv::self().parallel_size() << "... shutting down."
        << std::endl;
      throw std::runtime_error("shutdown");
    }
    serializedIOGroupSize_ = siogs;
    Ioss::SerializeIO::setGroupFactor(siogs);
  }
}

void
Simulation::breadboard()
{
  realms_->breadboard();
  timeIntegrator_->breadboard();
  transfers_->breadboard();
}

void
Simulation::initialize()
{
  realms_->initialize_prolog();
  timeIntegrator_->initialize();
  transfers_->initialize();
  realms_->initialize_epilog();
}

void
Simulation::init_prolog()
{
  realms_->initialize_prolog();
  timeIntegrator_->overset_->initialize();
}

void
Simulation::init_epilog()
{
  realms_->initialize_epilog();
  transfers_->initialize();
}

void
Simulation::run()
{
  KynemaUGFEnv::self().kynema_ugfOutputP0() << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "*******************************************************" << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "Simulation Shall Commence: number of processors = "
    << KynemaUGFEnv::self().parallel_size() << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "*******************************************************" << std::endl;

  timeIntegrator_->integrate_realm();
}

void
Simulation::high_level_banner()
{

  std::vector<std::string> additionalTPLs;
#ifdef KYNEMA_UGF_USES_FFTW
  additionalTPLs.push_back("FFTW");
#endif
#ifdef KYNEMA_UGF_USES_OPENFAST
  additionalTPLs.push_back("OpenFAST");
#endif
#ifdef KYNEMA_UGF_USES_HYPRE
  additionalTPLs.push_back("Hypre");
#endif
#ifdef KYNEMA_UGF_USES_TIOGA
  additionalTPLs.push_back("TIOGA");
#endif

  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "======================================================================="
       "========"
    << std::endl
    << "                                  Kynema-UGF                           "
       " "
       "        "
    << std::endl
    << "       An incompressible, turbulent computational fluid dynamics "
       "solver        "
    << std::endl
    << "                  for wind turbine and wind farm simulations           "
       "        "
    << std::endl
    << "======================================================================="
       "========"
    << std::endl
    << std::endl
    << "   Kynema-UGF Version: " << version::KynemaUGFVersionTag << std::endl
    << "   Kynema-UGF GIT Commit SHA: " << version::KynemaUGFGitCommitSHA
    << ((version::RepoIsDirty == "DIRTY") ? ("-" + version::RepoIsDirty) : "")
    << std::endl
    << "   Trilinos Version: " << version::TrilinosVersionTag << std::endl
    << std::endl
    << "   TPLs: Boost, HDF5, netCDF, STK, Trilinos, yaml-cpp and zlib   "
    << std::endl;

  if (additionalTPLs.size() > 0) {
    KynemaUGFEnv::self().kynema_ugfOutputP0() << "   Optional TPLs enabled: ";
    int numTPLs = additionalTPLs.size();
    for (int i = 0; i < (numTPLs - 1); i++)
      KynemaUGFEnv::self().kynema_ugfOutputP0() << additionalTPLs[i] << ", ";
    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << additionalTPLs[numTPLs - 1] << std::endl;
  }

  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "   Copyright 2017 National Technology & Engineering Solutions of "
       "Sandia, LLC   "
    << std::endl
    << "   (NTESS), National Renewable Energy Laboratory, University of Texas "
       "Austin,  "
    << std::endl
    << "    Northwest Research Associates. Under the terms of Contract "
       "DE-NA0003525    "
    << std::endl
    << "    with NTESS, the U.S. Government retains certain rights in this "
       "software.   "
    << std::endl
    << "                                                                       "
       "        "
    << std::endl
    << "           This software is released under the BSD 3-clause license.   "
       "        "
    << std::endl
    << "   See LICENSE file at https://github.com/exawind/kynema_ugf for more "
       "details.  "
    << std::endl
    << "-----------------------------------------------------------------------"
       "--------"
    << std::endl
    << std::endl;

  if (!std::is_same<DeviceSpace, Kokkos::Serial>::value) {
    // Save output from the master proc in the log file
    Kokkos::DefaultExecutionSpace{}.print_configuration(
      KynemaUGFEnv::self().kynema_ugfOutputP0());
    // But have everyone print out to standard error for debugging purposes
    Kokkos::DefaultExecutionSpace{}.print_configuration(std::cerr);
  }
}
} // namespace kynema_ugf
} // namespace sierra
