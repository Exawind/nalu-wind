// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <kernel/KernelBuilderLog.h>
#include <NaluEnv.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace sierra {
namespace nalu {
//--------------------------------------------------------------------------
KernelBuilderLog&
KernelBuilderLog::self()
{
  static KernelBuilderLog instance;
  return instance;
}
//--------------------------------------------------------------------------
void
KernelBuilderLog::add_valid_name(std::string kernelTypeName, std::string name)
{
  validKernelNames_[kernelTypeName].insert(name);
}
//--------------------------------------------------------------------------
void
KernelBuilderLog::add_built_name(std::string kernelTypeName, std::string name)
{
  builtKernelNames_[kernelTypeName].insert(name);
}
//--------------------------------------------------------------------------
bool
KernelBuilderLog::print_invalid_kernel_names(
  std::string kernelTypeName,
  const std::map<std::string, std::vector<std::string>>& srcTermsMap)
{
  auto it = srcTermsMap.find(kernelTypeName);
  if (it == srcTermsMap.end()) {
    return false;
  }
  const std::vector<std::string>& inputFileNames = it->second;
  const std::set<std::string> implementedKernelNames =
    valid_kernel_names(kernelTypeName);

  std::vector<std::string> badNames;
  for (const auto& inputFileName : inputFileNames) {
    if (
      implementedKernelNames.find(inputFileName) ==
      implementedKernelNames.end()) {
      badNames.push_back(inputFileName);
    }
  }

  for (const auto& name : badNames) {
    NaluEnv::self().naluOutputP0()
      << "Error: No Supplemental Algorithm with name `" << name
      << "' implemented" << std::endl;
  }

  bool isOK = badNames.empty();

  return isOK;
}
//--------------------------------------------------------------------------
void
KernelBuilderLog::print_valid_kernel_names(std::string kernelTypeName)
{
  const auto implementedKernelNames = valid_kernel_names(kernelTypeName);

  std::string msgList = "";
  for (const auto& name : implementedKernelNames) {
    msgList += "`" + name + "', ";
  }
  msgList = msgList.substr(0, msgList.size() - 2);

  NaluEnv::self().naluOutputP0()
    << "Valid Supplemental Algorithm names for " << kernelTypeName << " are "
    << msgList << "." << std::endl;
}
//--------------------------------------------------------------------------
void
KernelBuilderLog::print_built_kernel_names(std::string kernelTypeName)
{
  const auto builtKernelNames = built_kernel_names(kernelTypeName);

  std::string msgList = "";
  for (const auto& name : builtKernelNames) {
    msgList += "`" + name + "', ";
  }
  msgList = msgList.substr(0, msgList.size() - 2);

  NaluEnv::self().naluOutputP0() << "Built Kernels for " << kernelTypeName
                                 << " are " << msgList << "." << std::endl;
}
//--------------------------------------------------------------------------
std::set<std::string>
KernelBuilderLog::valid_kernel_names(std::string kernelTypeName)
{
  auto it = validKernelNames_.find(kernelTypeName);
  if (it == validKernelNames_.end()) {
    return std::set<std::string>();
  }
  return it->second;
}
//--------------------------------------------------------------------------
std::set<std::string>
KernelBuilderLog::built_kernel_names(std::string kernelTypeName)
{
  auto it = builtKernelNames_.find(kernelTypeName);
  if (it == builtKernelNames_.end()) {
    return std::set<std::string>();
  }
  return it->second;
}

} // namespace nalu
} // namespace sierra
