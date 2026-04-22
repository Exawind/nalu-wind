// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <mpi.h>
#include <stk_util/diag/PrintTimer.hpp>

// kynema_ugf
#include <KynemaUGFParsing.h>
#include <Simulation.h>
#include <KynemaUGFEnv.h>
#include <KynemaUGFVersionInfo.h>

// util
#include <stk_util/environment/perf_util.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

// input params
#include <stk_util/environment/OptionsSpecification.hpp>
#include <stk_util/environment/ParseCommandLineArgs.hpp>
#include <stk_util/environment/ParsedOptions.hpp>

// yaml for parsing..
#include <yaml-cpp/yaml.h>

// Kokkos
#include <Kokkos_Core.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "HypreNGP.h"

#include "master_element/MasterElementRepo.h"

static std::string
human_bytes_double(double bytes)
{
  const double K = 1024;
  const double M = K * 1024;
  const double G = M * 1024;

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

int
main(int argc, char** argv)
{
  namespace version = sierra::kynema_ugf::version;

  // start up MPI
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    throw std::runtime_error("MPI_Init failed");
  }

  // KynemaUGFEnv singleton
  sierra::kynema_ugf::KynemaUGFEnv& kynema_ugfEnv =
    sierra::kynema_ugf::KynemaUGFEnv::self();

  Kokkos::initialize(argc, argv);

  // Hypre initialization
  kynema_ugf_hypre::hypre_initialize();

  {

    stk::diag::setEnabledTimerMetricsMask(
      stk::diag::METRICS_CPU_TIME | stk::diag::METRICS_WALL_TIME);

    sierra::kynema_ugf::Simulation::rootTimer().start();

    // start initial time
    double start_time = kynema_ugfEnv.kynema_ugf_time();

    // command line options.
    std::string inputFileName, logFileName;
    bool debug = false;
    int serializedIOGroupSize = 0;
    const std::string kynema_ugfVersion =
      (version::RepoIsDirty == "DIRTY")
        ? (version::KynemaUGFVersionTag + "-dirty")
        : version::KynemaUGFVersionTag;

    stk::OptionsSpecification desc("KynemaUGF Supported Options");
    std::string kynema_ugfVout = kynema_ugfVersion.c_str();
    desc.add_options()("help,h", "Help message")(
      "version,v", kynema_ugfVersion.c_str())(
      "input-deck,i", "Analysis input file",
      stk::DefaultValue<std::string>("kynema_ugf.i"),
      stk::TargetPointer<std::string>(&inputFileName))(
      "log-file,o", "Analysis log file",
      stk::TargetPointer<std::string>(&logFileName))(
      "serialized-io-group-size,s",
      "Specifies the number of processors that can concurrently perform I/O. "
      "Specifying zero disables serialization.",
      stk::DefaultValue<int>(0),
      stk::TargetPointer<int>(&serializedIOGroupSize))(
      "debug,D", "Debug output to the log file")(
      "pprint,p", "Parallel output to the number of mpi rank log files ");

    stk::ParsedOptions parsedOptions;
    stk::parse_command_line_args(
      argc, const_cast<const char**>(argv), desc, parsedOptions);

    // deal with some default parameters
    if (parsedOptions.count("help")) {
      if (!kynema_ugfEnv.parallel_rank())
        std::cerr << desc << std::endl;
      return 0;
    }

    if (parsedOptions.count("version")) {
      if (!kynema_ugfEnv.parallel_rank())
        std::cerr << "Version: " << kynema_ugfVersion << std::endl;
      return 0;
    }

    if (parsedOptions.count("debug")) {
      debug = true;
    }

    std::ifstream fin(inputFileName.c_str());
    if (!fin.good()) {
      if (!kynema_ugfEnv.parallel_rank())
        std::cerr << "Input file is not specified or does not exist: user "
                     "specified (or default) name= "
                  << inputFileName << std::endl;
      return 0;
    }

    // deal with logfile name; if none supplied, go with inputFileName.log
    if (!parsedOptions.count("log-file")) {
      int dotPos = inputFileName.rfind(".");
      if (-1 == dotPos) {
        // lacking extension
        logFileName = inputFileName + ".log";
      } else {
        // with extension; swap with .log
        logFileName = inputFileName.substr(0, dotPos) + ".log";
      }
    }

    bool pprint = false;
    if (parsedOptions.count("pprint")) {
      pprint = true;
    }
    // deal with log file stream
    const bool capture_stdout = true;
    kynema_ugfEnv.set_log_file_stream(logFileName, pprint, capture_stdout);

    // proceed with reading input file "document" from YAML
    YAML::Node doc = YAML::LoadFile(inputFileName.c_str());
    if (!kynema_ugfEnv.parallel_rank()) {
      std::cout << std::string(20, '#') << " INPUT FILE START "
                << std::string(20, '#') << std::endl;
      sierra::kynema_ugf::KynemaUGFParsingHelper::emit(std::cout, doc);
      std::cout << std::string(20, '#') << " INPUT FILE END   "
                << std::string(20, '#') << std::endl;
    }

    // Hypre general parameter setting
    kynema_ugf_hypre::hypre_set_params(doc);

    sierra::kynema_ugf::Simulation sim(doc);
    if (serializedIOGroupSize) {
      kynema_ugfEnv.kynema_ugfOutputP0()
        << "Info: found non-zero serialized_io_group_size on command-line= "
        << serializedIOGroupSize << " (takes precedence over input file value)."
        << std::endl;
      sim.setSerializedIOGroupSize(serializedIOGroupSize);
    }
    kynema_ugfEnv.debug_ = debug;
    sim.load(doc);
    sim.breadboard();
    sim.initialize();
    sim.run();

    // stop timer
    const double stop_time = kynema_ugfEnv.kynema_ugf_time();
    const double total_time = stop_time - start_time;
    const char* timer_name = "Total Time";

    // parallel reduce overall times
    double g_sum, g_min, g_max;
    stk::all_reduce_min(kynema_ugfEnv.parallel_comm(), &total_time, &g_min, 1);
    stk::all_reduce_max(kynema_ugfEnv.parallel_comm(), &total_time, &g_max, 1);
    stk::all_reduce_sum(kynema_ugfEnv.parallel_comm(), &total_time, &g_sum, 1);
    const int nprocs = kynema_ugfEnv.parallel_size();

    // output total time
    kynema_ugfEnv.kynema_ugfOutputP0()
      << "Timing for Simulation: nprocs= " << nprocs << std::endl;
    kynema_ugfEnv.kynema_ugfOutputP0()
      << "           main() --  "
      << " \tavg: " << g_sum / double(nprocs) << " \tmin: " << g_min
      << " \tmax: " << g_max << std::endl;

    // output memory usage
    {
      size_t now, hwm;
      stk::get_memory_usage(now, hwm);
      // min, max, sum
      size_t global_now[3] = {now, now, now};
      size_t global_hwm[3] = {hwm, hwm, hwm};

      stk::all_reduce(
        kynema_ugfEnv.parallel_comm(), stk::ReduceSum<1>(&global_now[2]));
      stk::all_reduce(
        kynema_ugfEnv.parallel_comm(), stk::ReduceMin<1>(&global_now[0]));
      stk::all_reduce(
        kynema_ugfEnv.parallel_comm(), stk::ReduceMax<1>(&global_now[1]));

      stk::all_reduce(
        kynema_ugfEnv.parallel_comm(), stk::ReduceSum<1>(&global_hwm[2]));
      stk::all_reduce(
        kynema_ugfEnv.parallel_comm(), stk::ReduceMin<1>(&global_hwm[0]));
      stk::all_reduce(
        kynema_ugfEnv.parallel_comm(), stk::ReduceMax<1>(&global_hwm[1]));

      kynema_ugfEnv.kynema_ugfOutputP0() << "Memory Overview: " << std::endl;

      kynema_ugfEnv.kynema_ugfOutputP0()
        << "kynema_ugf memory: total (over all cores) current/high-water mark= "
        << std::setw(15) << human_bytes_double(global_now[2]) << std::setw(15)
        << human_bytes_double(global_hwm[2]) << std::endl;

      kynema_ugfEnv.kynema_ugfOutputP0()
        << "kynema_ugf memory:   min (over all cores) current/high-water mark= "
        << std::setw(15) << human_bytes_double(global_now[0]) << std::setw(15)
        << human_bytes_double(global_hwm[0]) << std::endl;

      kynema_ugfEnv.kynema_ugfOutputP0()
        << "kynema_ugf memory:   max (over all cores) current/high-water mark= "
        << std::setw(15) << human_bytes_double(global_now[1]) << std::setw(15)
        << human_bytes_double(global_hwm[1]) << std::endl;
    }

    sierra::kynema_ugf::Simulation::rootTimer().stop();

    // output timings consistent w/ rest of Sierra
    stk::diag::Timer& sierra_timer =
      sierra::kynema_ugf::Simulation::rootTimer();
    const double elapsed_time =
      sierra_timer.getMetric<stk::diag::WallTime>().getAccumulatedLap(false);
    stk::diag::Timer& mesh_output_timer =
      sierra::kynema_ugf::Simulation::outputTimer();
    double mesh_output_time =
      mesh_output_timer.getMetric<stk::diag::WallTime>().getAccumulatedLap(
        false);
    double time_without_output = elapsed_time - mesh_output_time;

    stk::parallel_print_time_without_output_and_hwm(
      kynema_ugfEnv.parallel_comm(), time_without_output,
      kynema_ugfEnv.kynema_ugfOutputP0());

    if (!kynema_ugfEnv.parallel_rank())
      stk::print_timers_and_memory(&timer_name, &total_time, 1 /*num timers*/);

    stk::diag::printTimersTable(
      kynema_ugfEnv.kynema_ugfOutputP0(),
      sierra::kynema_ugf::Simulation::rootTimer(),
      stk::diag::METRICS_CPU_TIME | stk::diag::METRICS_WALL_TIME, false,
      kynema_ugfEnv.parallel_comm());

    stk::diag::deleteRootTimer(sierra::kynema_ugf::Simulation::rootTimer());

    // Write out Trilinos timers
    Teuchos::TimeMonitor::summarize(
      kynema_ugfEnv.kynema_ugfOutputP0(), false, true, false, Teuchos::Union);

    // Master element cleanup
    sierra::kynema_ugf::MasterElementRepo::clear();
  }

  // Hypre cleanup
  kynema_ugf_hypre::hypre_finalize();

  Kokkos::finalize();

  MPI_Finalize();

  // all done
  return 0;
}
