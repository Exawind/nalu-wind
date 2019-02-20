/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <Simulation.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realms.h>
#include <xfer/Transfers.h>
#include <TimeIntegrator.h>
#include <LinearSolvers.h>
#include <NaluVersionInfo.h>

#include <Ioss_SerializeIO.h>

namespace sierra{
namespace nalu{

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
bool Simulation::debug_ = false;

Simulation::Simulation(const YAML::Node& root_node) :
    m_root_node(root_node),
    timeIntegrator_(NULL),
    realms_(NULL),
    transfers_(NULL),
    linearSolvers_(NULL),
    serializedIOGroupSize_(0)
{}

Simulation::~Simulation() {
  delete realms_;
  delete transfers_;
  delete timeIntegrator_;
  delete linearSolvers_;
}

// Timers
//static
stk::diag::TimerSet &
Simulation::rootTimerSet()
{
  static stk::diag::TimerSet s_timerSet(sierra::Diag::TIMER_ALL);

  return s_timerSet;
}

//static
stk::diag::Timer& Simulation::rootTimer()
{
  static stk::diag::Timer s_timer = stk::diag::createRootTimer("Nalu", rootTimerSet());

  return s_timer;
}

//static
stk::diag::Timer& Simulation::outputTimer()
{
  static stk::diag::Timer s_timer("Output", rootTimer());
  return s_timer;
}


void Simulation::load(const YAML::Node & node)
{

  high_level_banner();

  // load the linear solver configs
  linearSolvers_ = new LinearSolvers(*this);
  linearSolvers_->load(node);

  // create the realms
  realms_ = new Realms(*this);
  realms_->load(node);

  // create the time integrator
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "Time Integrator Review:  " << std::endl;
  NaluEnv::self().naluOutputP0() << "=========================" << std::endl;
  timeIntegrator_ = new TimeIntegrator(this);
  timeIntegrator_->load(node);

  // create the transfers; mesh is already loaded in realm
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "Transfer Review:         " << std::endl;
  NaluEnv::self().naluOutputP0() << "=========================" << std::endl;
  transfers_ = new Transfers(*this);
  transfers_->load(node);

}

void Simulation::setSerializedIOGroupSize(int siogs)
{
  if (siogs)
    {
      if (siogs < 0 || siogs > NaluEnv::self().parallel_size() || NaluEnv::self().parallel_size() % siogs != 0)
        {
          NaluEnv::self().naluOutputP0() << "Error: Job requested serialized_io_group_size of " << siogs
                          << " which is incompatible with MPI size= " << NaluEnv::self().parallel_size()
                          << "... shutting down." << std::endl;
          throw std::runtime_error("shutdown");
        }
      serializedIOGroupSize_ = siogs;
      Ioss::SerializeIO::setGroupFactor(siogs);
    }
}

void Simulation::breadboard()
{
  realms_->breadboard();
  timeIntegrator_->breadboard();
  transfers_->breadboard();
}
void Simulation::initialize()
{
  realms_->initialize();
  timeIntegrator_->initialize();
  transfers_->initialize();
}
void Simulation::run()
{
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "*******************************************************" << std::endl;
  NaluEnv::self().naluOutputP0() << "Simulation Shall Commence: number of processors = " << NaluEnv::self().parallel_size() << std::endl;
  NaluEnv::self().naluOutputP0() << "*******************************************************" << std::endl;

  timeIntegrator_->integrate_realm();
}

void Simulation::high_level_banner() {

  std::vector<std::string> additionalTPLs;
#ifdef NALU_USES_FFTW
  additionalTPLs.push_back("FFTW");
#endif
#ifdef NALU_USES_OPENFAST
  additionalTPLs.push_back("OpenFAST");
#endif
#ifdef NALU_USES_HYPRE
  additionalTPLs.push_back("Hypre");
#endif
#ifdef NALU_USES_TIOGA
  additionalTPLs.push_back("TIOGA");
#endif
#ifdef NALU_USES_PERCEPT
  additionalTPLs.push_back("Percept");
#endif

  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "=================================================================" << std::endl;
  NaluEnv::self().naluOutputP0() << "                            Nalu:                                " << std::endl;
  NaluEnv::self().naluOutputP0() << "      A low Mach number, turbulent reacting flow code with       " << std::endl;
  NaluEnv::self().naluOutputP0() << "            coupling to PMR and object response (CHT)            " << std::endl;
  NaluEnv::self().naluOutputP0() << "=================================================================" << std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0()
    << "   Nalu Version: " << version::NaluVersionTag << std::endl
    << "   Nalu GIT Commit SHA: " << version::NaluGitCommitSHA 
    << ((version::RepoIsDirty == "DIRTY") ? ("-" + version::RepoIsDirty) : "") << std::endl
    << "   Trilinos Version: " << version::TrilinosVersionTag << std::endl << std::endl;
  NaluEnv::self().naluOutputP0() << "   TPLs: Boost, HDF5, netCDF, STK, Trilinos, YAML_cpp and zlib   " << std::endl;

  if (additionalTPLs.size() > 0) {
    NaluEnv::self().naluOutputP0() << "   Optional TPLs enabled: ";
    int numTPLs = additionalTPLs.size();
    for (int i=0; i < (numTPLs - 1); i++)
      NaluEnv::self().naluOutputP0() << additionalTPLs[i] << ", ";
    NaluEnv::self().naluOutputP0() << additionalTPLs[numTPLs -1] << std::endl;
  }

  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "              Copyright 2014 Sandia Corporation.                 " << std::endl;
  NaluEnv::self().naluOutputP0() << "      This software is released under the license detailed       " << std::endl;
  NaluEnv::self().naluOutputP0() << "   in the file, LICENSE, which is located in the top-level Nalu  " << std::endl;
  NaluEnv::self().naluOutputP0() << "                     directory structure                         " << std::endl;
  NaluEnv::self().naluOutputP0() << "-----------------------------------------------------------------" << std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;

}
} // namespace nalu
} // namespace Sierra
