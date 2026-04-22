// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <KynemaUGFEnv.h>

#include <mpi.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>

#include <stk_util/environment/WallTime.hpp>

namespace sierra {
namespace kynema_ugf {

//==========================================================================
// Class Definition
//==========================================================================
// KynemaUGFEnv - manage parallel and parallel output in KynemaUGF
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KynemaUGFEnv::KynemaUGFEnv()
  : parallelCommunicator_(MPI_COMM_WORLD),
    pSize_(-1),
    pRank_(-1),
    stdoutStream_(std::cout.rdbuf()),
    kynema_ugfLogStream_(new std::ostream(std::cout.rdbuf())),
    kynema_ugfParallelStream_(
      new std::ostream(&kynema_ugfParallelStreamBuffer_)),
    parallelLog_(false),
    debug_(false)
{
  // initialize
  MPI_Comm_size(parallelCommunicator_, &pSize_);
  MPI_Comm_rank(parallelCommunicator_, &pRank_);
}

//--------------------------------------------------------------------------
//-------- self ------------------------------------------------------------
//--------------------------------------------------------------------------
KynemaUGFEnv&
KynemaUGFEnv::self()
{
  static KynemaUGFEnv s;
  return s;
}

//--------------------------------------------------------------------------
//-------- kynema_ugfOutputP0
//----------------------------------------------------
//--------------------------------------------------------------------------
std::ostream&
KynemaUGFEnv::kynema_ugfOutputP0()
{
  return *kynema_ugfLogStream_;
}

//--------------------------------------------------------------------------
//-------- kynema_ugfOutput
//------------------------------------------------------
//--------------------------------------------------------------------------
std::ostream&
KynemaUGFEnv::kynema_ugfOutput()
{
  return *kynema_ugfParallelStream_;
}

//--------------------------------------------------------------------------
//-------- parallel_size ---------------------------------------------------
//--------------------------------------------------------------------------
int
KynemaUGFEnv::parallel_size()
{
  return pSize_;
}

//--------------------------------------------------------------------------
//-------- parallel_rank ---------------------------------------------------
//--------------------------------------------------------------------------
int
KynemaUGFEnv::parallel_rank()
{
  return pRank_;
}

//--------------------------------------------------------------------------
//-------- parallel_comm ---------------------------------------------------
//--------------------------------------------------------------------------
MPI_Comm
KynemaUGFEnv::parallel_comm()
{
  return parallelCommunicator_;
}

//--------------------------------------------------------------------------
//-------- set_log_file_stream ---------------------------------------------
//--------------------------------------------------------------------------
void
KynemaUGFEnv::set_log_file_stream(
  std::string kynema_ugfLogName, bool pprint, const bool capture_cout)
{
  if (pRank_ == 0) {
    kynema_ugfStreamBuffer_.open(kynema_ugfLogName.c_str(), std::ios::out);
    kynema_ugfLogStream_->rdbuf(&kynema_ugfStreamBuffer_);
  } else {
    kynema_ugfLogStream_->rdbuf(&kynema_ugfEmptyStreamBuffer_);
  }

  if (capture_cout)
    std::cout.rdbuf(kynema_ugfLogStream_->rdbuf());

  // default to an empty stream buffer for parallel unless pprint is set
  parallelLog_ = pprint;
  if (parallelLog_) {
    int numPlaces = static_cast<int>(std::log10(pSize_ - 1) + 1);

    std::stringstream paddedRank;
    paddedRank << std::setw(numPlaces) << std::setfill('0') << parallel_rank();

    // inputname.log -> inputname.log.16.02 for the rank 2 proc of a 16 proc job
    std::string parallelLogName =
      kynema_ugfLogName + "." + std::to_string(pSize_) + "." + paddedRank.str();
    kynema_ugfParallelStreamBuffer_.open(
      parallelLogName.c_str(), std::ios::out);
    kynema_ugfParallelStream_->rdbuf(&kynema_ugfParallelStreamBuffer_);
  } else {
    kynema_ugfParallelStream_->rdbuf(stdoutStream_);
  }
}

//--------------------------------------------------------------------------
//-------- close_log_file_stream -------------------------------------------
//--------------------------------------------------------------------------
void
KynemaUGFEnv::close_log_file_stream()
{
  if (pRank_ == 0) {
    kynema_ugfStreamBuffer_.close();
  }
  if (parallelLog_) {
    kynema_ugfParallelStreamBuffer_.close();
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
KynemaUGFEnv::~KynemaUGFEnv()
{
  close_log_file_stream();
  delete kynema_ugfLogStream_;
  delete kynema_ugfParallelStream_;

  // shut down MPI
  // MPI_Finalize();
}

//--------------------------------------------------------------------------
//-------- kynema_ugf_time
//-------------------------------------------------------
//--------------------------------------------------------------------------
double
KynemaUGFEnv::kynema_ugf_time()
{
  return stk::wall_time();
}

} // namespace kynema_ugf
} // namespace sierra
