// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "user_functions/StringTimeCoordFunction.h"
#include "stk_expreval/Evaluator.hpp"

#include <stdexcept>

namespace sierra::nalu {

StringTimeCoordFunction::StringTimeCoordFunction(std::string fcn)
{
  stk::expreval::Eval eval;

  if (fcn.empty()) {
    throw std::runtime_error("Empty string function");
  }

  fcn.erase(std::remove_if(fcn.begin(), fcn.end(), ::isspace), fcn.end());
  std::transform(fcn.begin(), fcn.end(), fcn.begin(), ::toupper);

  try {
    eval.parse(fcn);
  } catch (std::exception& e) {
    std::ostringstream os;
    os << "unable to parse input string '" << fcn << "'" << std::endl;
    throw std::runtime_error(os.str());
  }

  if (eval.undefinedFunction()) {
    std::ostringstream os;
    os << "found an undefined function in '" << fcn << "'" << std::endl;
    throw std::runtime_error(os.str());
  }

  auto bind_index = [&eval](const char* name) {
    const auto& names = eval.get_independent_variable_names();
    if (std::find(names.begin(), names.end(), name) != names.end()) {
      return eval.get_variable_index(name);
    }
    return fcn::UNMAPPED_INDEX;
  };

  t_index = bind_index("T");
  x_index = bind_index("X");
  y_index = bind_index("Y");
  z_index = bind_index("Z");

  std::vector<std::string> valid_entries{"T", "X", "Y", "Z"};
  for (const auto& name : eval.get_independent_variable_names()) {
    if (
      std::find(valid_entries.begin(), valid_entries.end(), name) ==
      valid_entries.end()) {
      throw std::runtime_error(
        "Invalid input name in DeviceStringFunction " + name);
    }
  }

  constant = eval.is_constant_expression();
  parsed_eval = eval.get_parsed_eval();
  try {
    eval.evaluate();
    (*this)(0, 0, 0, 0);
  } catch (std::exception& e) {
    std::ostringstream sout;
    sout << "String function was unable to be evaluated with input string '"
         << fcn << "'.\n"
         << e.what();
    throw std::runtime_error(sout.str());
  }
}

template <int N>
KOKKOS_FUNCTION void
bind_if_valid(
  stk::expreval::DeviceVariableMap<N>& var_map, int index, double val)
{
  if (index >= 0 && index < N) {
    var_map[index] = val;
  }
}

KOKKOS_FUNCTION double
StringTimeCoordFunction::operator()(
  double t, double x, double y, double z) const
{
  constexpr int num_vars = 4;
  auto var_map = stk::expreval::DeviceVariableMap<num_vars>(parsed_eval);
  bind_if_valid(var_map, t_index, t);
  bind_if_valid(var_map, x_index, x);
  bind_if_valid(var_map, y_index, y);
  bind_if_valid(var_map, z_index, z);
  return parsed_eval.evaluate(var_map);
}

} // namespace sierra::nalu
