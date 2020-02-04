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

#include<Kokkos_Core.hpp>
#include<Kokkos_DualView.hpp>
#include<actuator/ActuatorTypes.h>

namespace sierra{
namespace nalu{


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
 *  (i.e. preIter vs findPoints)
 */

template<typename BulkData, typename TAG, typename ExecutionSpace>
struct ActuatorFunctor{
  // Kokkos magic
  // set execution space by template parameter (overrides default)
  using execution_space = ExecutionSpace;
  // define templated execution space's matching memory space
  using memory_space = typename std::conditional<
      std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
      Kokkos::DualView<double*>::memory_space, Kokkos::DualView<double*>::host_mirror_space>::type;

  using ActVectorDbl =
      const Kokkos::View<ActVectorDblDv::scalar_array_type,
      ActVectorDblDv::array_layout,
      memory_space>;

  BulkData& bulk_;
  ActuatorFunctor(BulkData& bulk);
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const;
};

template<typename BulkData>
struct ActuatorComputePointLocation{
  BulkData& bulk_;
  ActuatorComputePointLocation(BulkData& bulk):bulk_(bulk){}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const;
};

template<typename BulkData>
struct ActuatorInterpolateFieldValues{
  BulkData& bulk_;
  ActuatorInterpolateFieldValues(BulkData& bulk):bulk_(bulk){}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const;
};

template<typename BulkData>
struct ActuatorSpreadForces{
  BulkData& bulk_;
  ActuatorSpreadForces(BulkData& bulk):bulk_(bulk){}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const;
};

template<typename BulkData>
struct ActuatorPostIteration{
  BulkData& bulk_;
  ActuatorPostIteration(BulkData& bulk):bulk_(bulk){}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const;
};

/*!
 * \class Actuator
 * \brief Template class for implementing Actuator execution
 *
 * This class allows one to create an actuator execution model
 * that can be varied based on the Meta and Bulk data types supplied.
 * Data extents and parameters should be passed via meta data and
 * memory allocation should occur during the constructor of this class.
 *
 * Specific details of the implementation are done via the execute() method.
 *
 * \tparam MetaData Container holding data params and extents
 *
 * \tparam BulkData Container holding actual fields and additional objects i.e. FAST
 */
template<typename MetaData, typename BulkData>
class Actuator
{
public:
  Actuator(MetaData actMeta);
  // TODO(psakiev) restrict access for this except for unit testing
  const BulkData& actuator_bulk(){return actBulk_;}
  /// Where the work is done. This function should be defined for each particular instance
  void execute();

private:
  BulkData actBulk_; //< Contains data and a copy of the meta data that was used in construction

};

}
}
#endif
