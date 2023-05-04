// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NaluEnv_h
#define NaluEnv_h

#include <mpi.h>
#include <fstream>
#include <streambuf>

namespace sierra {
namespace nalu {

class NaluEmptyStreamBuffer : public std::filebuf
{
public:
  int overflow(int c) { return c; }
};

class NaluEnv
{
public:
  NaluEnv();
  ~NaluEnv();

  static NaluEnv& self();
  MPI_Comm parallelCommunicator_;
  int pSize_;
  int pRank_;
  std::streambuf* stdoutStream_;
  std::ostream* naluLogStream_;
  std::ostream* naluParallelStream_;
  bool parallelLog_;

  NaluEmptyStreamBuffer naluEmptyStreamBuffer_;
  std::filebuf naluStreamBuffer_;
  std::filebuf naluParallelStreamBuffer_;
  bool debug_;

  std::ostream& naluOutputP0();
  std::ostream& naluOutput();

  MPI_Comm parallel_comm();
  int parallel_size();
  int parallel_rank();
  bool debug() { return debug_; }

  /** Redirect output to a log file
   *
   *  \param naluLogName Name of the file where outputs are redirected
   *
   *  \param pprint (Parallel print) If true, all MPI ranks output to their own
   * file
   *
   *  \param capture_cout If true, `std::cout` is redirected to log file
   *
   */
  void set_log_file_stream(
    std::string naluLogName,
    bool pprint = false,
    const bool capture_cout = false);
  void close_log_file_stream();
  double nalu_time();
};

} // namespace nalu
} // namespace sierra

#endif
