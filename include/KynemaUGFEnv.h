// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef KynemaUGFEnv_h
#define KynemaUGFEnv_h

#include <mpi.h>
#include <fstream>
#include <streambuf>

namespace sierra {
namespace kynema_ugf {

class KynemaUGFEmptyStreamBuffer : public std::filebuf
{
public:
  int overflow(int c) { return c; }
};

class KynemaUGFEnv
{
public:
  KynemaUGFEnv();
  ~KynemaUGFEnv();

  static KynemaUGFEnv& self();
  MPI_Comm parallelCommunicator_;
  int pSize_;
  int pRank_;
  std::streambuf* stdoutStream_;
  std::ostream* kynema_ugfLogStream_;
  std::ostream* kynema_ugfParallelStream_;
  bool parallelLog_;

  KynemaUGFEmptyStreamBuffer kynema_ugfEmptyStreamBuffer_;
  std::filebuf kynema_ugfStreamBuffer_;
  std::filebuf kynema_ugfParallelStreamBuffer_;
  bool debug_;

  std::ostream& kynema_ugfOutputP0();
  std::ostream& kynema_ugfOutput();

  MPI_Comm parallel_comm();
  int parallel_size();
  int parallel_rank();
  bool debug() { return debug_; }

  /** Redirect output to a log file
   *
   *  \param kynema-ugfLogName Name of the file where outputs are redirected
   *
   *  \param pprint (Parallel print) If true, all MPI ranks output to their own
   * file
   *
   *  \param capture_cout If true, `std::cout` is redirected to log file
   *
   */
  void set_log_file_stream(
    std::string kynema_ugfLogName,
    bool pprint = false,
    const bool capture_cout = false);
  void close_log_file_stream();
  double kynema_ugf_time();
};

} // namespace kynema_ugf
} // namespace sierra

#endif
