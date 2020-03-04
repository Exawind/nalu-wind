// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATOR_NGP_H_
#define ACTUATOR_NGP_H_

#include <actuator/ActuatorTypes.h>
#include <stk_mesh/base/BulkData.hpp>

namespace sierra {
namespace nalu {

/*! \struct AcutatorFunctor
 *  \brief Generalized functor for performing actuator work
 *
 * The goal of this functor is to generalize the interface
 * for doing work on host/device
 *
 * \tparam BulkData The type of bulk data that this functor will execute on
 *  (i.e. Actuator line w/h FAST vs a generalized instance)
 *
 * \tparam TAG A TAG to distinguish instances of operations
 *  (i.e. preIter vs findPoints) a tag is selected so the
 *  functors don't have to be predefined and/or constructed
 *  in the Actuator class.  Rather, their constructor can be
 *  called in the Kokkos::parallel_for
 */

template <typename BulkData, typename TAG, typename ExecutionSpace>
struct ActuatorFunctor
{
  // Kokkos magic
  // set execution space by template parameter (overrides default)
  using execution_space = ExecutionSpace;
  // define templated execution space's matching memory space
  using memory_space = typename std::conditional<
    std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
    Kokkos::DualView<double*>::memory_space,
    Kokkos::HostSpace>::type;

  BulkData& actBulk_;
  ActuatorFunctor(BulkData& bulk);

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const;

  template<typename T>
  KOKKOS_INLINE_FUNCTION
  auto get_local_view(T dualView) const->decltype (dualView.template view<memory_space>()) {
    return dualView.template view<memory_space>();
  }

  template<typename T>
  KOKKOS_INLINE_FUNCTION
  void touch_dual_view(T dualView){
    dualView.template sync<memory_space>();
    dualView.template modify<memory_space>();
  }

};


/*!
 * \class Actuator
 * \brief Template class for implementing Actuator execution
 *
 * This class allows one to create an actuator execution model
 * that can be varied based on the ActuatorMeta and ActuatorBulk data types
 * supplied. Data extents and parameters should be passed via meta data and
 * memory allocation should occur during the constructor of this class.
 *
 * Specific details of the implementation are done via the execute() method.
 *
 * \tparam MetaData Container holding data params and extents
 *
 * \tparam BulkData Container holding actual fields and additional objects i.e.
 * FAST
 */
template <typename ActMetaData, typename ActBulkData>
class ActuatorNGP
{
public:
  ActuatorNGP(const ActMetaData& actMeta, stk::mesh::BulkData& stkBulk)
    : actMeta_(actMeta),
      actBulk_(actMeta_, stkBulk),
      numActPoints_(actBulk_.totalNumPoints_)
  {
  }
  // TODO(psakiev) restrict access for this except for unit testing
  const ActBulkData& actuator_bulk() { return actBulk_; }
  /// Where the work is done. This function should be defined for each
  /// particular instance
  void execute();

private:
  const ActMetaData actMeta_; //< Contains meta data used to construct
  ActBulkData actBulk_;       //< Contains data
  const int numActPoints_;    //< Total number of actuator points
};

} // namespace nalu
} // namespace sierra
#endif
