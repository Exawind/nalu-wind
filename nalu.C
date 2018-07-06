/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <mpi.h>
#include <stk_util/diag/PrintTimer.hpp>

// nalu
#include <NaluParsing.h>
#include <Simulation.h>
#include <NaluEnv.h>
#include <NaluVersionInfo.h>

// util
#include <stk_util/environment/perf_util.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

// boost for input params
#include <boost/program_options.hpp>

// yaml for parsing..
#include <yaml-cpp/yaml.h>

// Kokkos
#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

static std::string human_bytes_double(double bytes)
{
  const double K = 1024;
  const double M = K*1024;
  const double G = M*1024;

  std::ostringstream out;
  if (bytes < K) {
    out << bytes << " B";
  } else if (bytes < M) {
    bytes /= K;
    out << bytes << " K";
  } else if (bytes < G) {
    bytes /= M;
    out << bytes << " M";
  } else {
    bytes /= G;
    out << bytes << " G";
  }
  return out.str();
}


int main( int argc, char ** argv )
{
  namespace version = sierra::nalu::version;

  // start up MPI
  if ( MPI_SUCCESS != MPI_Init( &argc , &argv ) ) {
    throw std::runtime_error("MPI_Init failed");
  }

  // NaluEnv singleton
  sierra::nalu::NaluEnv &naluEnv = sierra::nalu::NaluEnv::self();
  Kokkos::initialize(argc, argv);
  {
  
  stk::diag::setEnabledTimerMetricsMask(stk::diag::METRICS_CPU_TIME | stk::diag::METRICS_WALL_TIME);

  sierra::nalu::Simulation::rootTimer().start();

  // start initial time
  double start_time = naluEnv.nalu_time();

  // command line options.
  std::string inputFileName, logFileName;
  bool debug = false;
  int serializedIOGroupSize = 0;
  const std::string naluVersion = (version::RepoIsDirty == "DIRTY")
    ? (version::NaluVersionTag + "-dirty")
    : version::NaluVersionTag;

  boost::program_options::options_description desc("Nalu Supported Options");
  desc.add_options()
    ("help,h","Help message")
    ("version,v", naluVersion.c_str())
    ("input-deck,i", boost::program_options::value<std::string>(&inputFileName)->default_value("nalu.i"),
        "Analysis input file")
    ("log-file,o", boost::program_options::value<std::string>(&logFileName),
        "Analysis log file")
    ("serialized-io-group-size,s",
     boost::program_options::value<int>(&serializedIOGroupSize)->default_value(0),
        "Specifies the number of processors which can concurrently perform I/O. Specifying zero disables serialization.")
    ("debug,D", "debug print on")
    ("pprint,p", "parallel print on");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

  boost::program_options::notify(vm);

  // deal with some default parameters
  if ( vm.count("help") ) {
    if (!naluEnv.parallel_rank())
      std::cerr << desc << std::endl;
    return 0;
  }

  if (vm.count("version")) {
    if (!naluEnv.parallel_rank())
      std::cerr << "Version: " << naluVersion << std::endl;
    return 0;
  }

  if (vm.count("debug")) {
    debug = true;
  }

  std::ifstream fin(inputFileName.c_str());
  if (!fin.good()) {
    if (!naluEnv.parallel_rank())
      std::cerr << "Input file is not specified or does not exist: user specified (or default) name= " << inputFileName << std::endl;
    return 0;
  }

  // deal with logfile name; if none supplied, go with inputFileName.log
  if (!vm.count("log-file")) {
    int dotPos = inputFileName.rfind(".");
    if ( -1 == dotPos ) {  
      // lacking extension
      logFileName = inputFileName + ".log";
    } 
    else {  
      // with extension; swap with .log
      logFileName = inputFileName.substr(0, dotPos) + ".log";
    }
  }

  bool pprint = false;
  if (vm.count("pprint")) {
    pprint = true;
  }
  // deal with log file stream
  naluEnv.set_log_file_stream(logFileName, pprint);

  // proceed with reading input file "document" from YAML
  YAML::Node doc = YAML::LoadFile(inputFileName.c_str());
  if (debug) {
    if (!naluEnv.parallel_rank())
      sierra::nalu::NaluParsingHelper::emit(std::cout, doc);
  }

  sierra::nalu::Simulation sim(doc);
  if (serializedIOGroupSize) {
    naluEnv.naluOutputP0() << "Info: found non-zero serialized_io_group_size on command-line= "
        << serializedIOGroupSize << " (takes precedence over input file value)."
        << std::endl;
    sim.setSerializedIOGroupSize(serializedIOGroupSize);
  }
  sim.debug_ = debug;
  sim.load(doc);
  sim.breadboard();
  sim.initialize();
  sim.run();

  // stop timer
  const double stop_time = naluEnv.nalu_time();
  const double total_time = stop_time - start_time;
  const char* timer_name = "Total Time";

  // parallel reduce overall times
  double g_sum, g_min, g_max;
  stk::all_reduce_min(naluEnv.parallel_comm(), &total_time, &g_min, 1);
  stk::all_reduce_max(naluEnv.parallel_comm(), &total_time, &g_max, 1);
  stk::all_reduce_sum(naluEnv.parallel_comm(), &total_time, &g_sum, 1);
  const int nprocs = naluEnv.parallel_size();

  // output total time
  naluEnv.naluOutputP0() << "Timing for Simulation: nprocs= " << nprocs << std::endl;
  naluEnv.naluOutputP0() << "           main() --  " << " \tavg: " << g_sum/double(nprocs)
			 << " \tmin: " << g_min << " \tmax: " << g_max << std::endl;

  // output memory usage
  {
    size_t now, hwm;
    stk::get_memory_usage(now, hwm);
    // min, max, sum
    size_t global_now[3] = {now,now,now};
    size_t global_hwm[3] = {hwm,hwm,hwm};

    stk::all_reduce(naluEnv.parallel_comm(), stk::ReduceSum<1>( &global_now[2] ) );
    stk::all_reduce(naluEnv.parallel_comm(), stk::ReduceMin<1>( &global_now[0] ) );
    stk::all_reduce(naluEnv.parallel_comm(), stk::ReduceMax<1>( &global_now[1] ) );

    stk::all_reduce(naluEnv.parallel_comm(), stk::ReduceSum<1>( &global_hwm[2] ) );
    stk::all_reduce(naluEnv.parallel_comm(), stk::ReduceMin<1>( &global_hwm[0] ) );
    stk::all_reduce(naluEnv.parallel_comm(), stk::ReduceMax<1>( &global_hwm[1] ) );

    naluEnv.naluOutputP0() << "Memory Overview: " << std::endl;

    naluEnv.naluOutputP0() << "nalu memory: total (over all cores) current/high-water mark= "
                            << std::setw(15) << human_bytes_double(global_now[2])
                            << std::setw(15) << human_bytes_double(global_hwm[2])
                            << std::endl;

    naluEnv.naluOutputP0() << "nalu memory:   min (over all cores) current/high-water mark= "
                            << std::setw(15) << human_bytes_double(global_now[0])
                            << std::setw(15) << human_bytes_double(global_hwm[0])
                            << std::endl;

    naluEnv.naluOutputP0() << "nalu memory:   max (over all cores) current/high-water mark= "
                            << std::setw(15) << human_bytes_double(global_now[1])
                            << std::setw(15) << human_bytes_double(global_hwm[1])
                            << std::endl;
  }

  sierra::nalu::Simulation::rootTimer().stop();

  //output timings consistent w/ rest of Sierra
  stk::diag::Timer & sierra_timer = sierra::nalu::Simulation::rootTimer();
  const double elapsed_time = sierra_timer.getMetric<stk::diag::WallTime>().getAccumulatedLap(false);
  stk::diag::Timer & mesh_output_timer = sierra::nalu::Simulation::outputTimer();
  double mesh_output_time = mesh_output_timer.getMetric<stk::diag::WallTime>().getAccumulatedLap(false);
  double time_without_output = elapsed_time-mesh_output_time;

  stk::parallel_print_time_without_output_and_hwm(naluEnv.parallel_comm(), time_without_output, naluEnv.naluOutputP0());

  if (!naluEnv.parallel_rank())
    stk::print_timers_and_memory(&timer_name, &total_time, 1 /*num timers*/);

  stk::diag::printTimersTable(naluEnv.naluOutputP0(), sierra::nalu::Simulation::rootTimer(),
                              stk::diag::METRICS_CPU_TIME | stk::diag::METRICS_WALL_TIME,
                              false, naluEnv.parallel_comm());

  stk::diag::deleteRootTimer(sierra::nalu::Simulation::rootTimer());
  }
  Kokkos::finalize_all();
  // all done  
  return 0;
}
