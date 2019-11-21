// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef KernelBuilderLog_h
#define KernelBuilderLog_h

#include <map>
#include <set>
#include <string>
#include <vector>

namespace sierra{
namespace nalu{
  class KernelBuilderLog
  {
  public:
    static KernelBuilderLog& self();
    KernelBuilderLog(const KernelBuilderLog&) = delete;
    void operator=(const KernelBuilderLog&) = delete;

    void add_valid_name(std::string kernelTypeName, std::string name);
    void add_built_name(std::string kernelTypeName, std::string name);

    bool print_invalid_kernel_names(
      std::string kernelTypeName,
      const std::map<std::string, std::vector<std::string>>& inputFileNames );

    void print_valid_kernel_names(std::string kernelTypeName);
    void print_built_kernel_names(std::string kernelTypeName);

    std::set<std::string> valid_kernel_names(std::string kernelTypeName);
    std::set<std::string> built_kernel_names(std::string kernelTypeName);
  private:
    KernelBuilderLog() = default;

    std::map<std::string, std::set<std::string>> validKernelNames_;
    std::map<std::string, std::set<std::string>> builtKernelNames_;
  };

} // namespace nalu
} // namespace Sierra

#endif
