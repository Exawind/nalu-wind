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

// Kokkos compatible functors that can be implemented based on
// BulkData Type

/*
 * The goal of this
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
      Kokkos::DualView<ActVectorDblDv::scalar_array_type,
      ActVectorDblDv::array_layout,
      memory_space>;

  BulkData& bulk_;
  ActuatorFunctor(BulkData& bulk);//:bulk_(bulk){}
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


template<typename MetaData, typename BulkData>
class Actuator
{
public:
  Actuator(MetaData actMeta);
  const BulkData& actuator_bulk(){return actBulk_;}
  void execute();

private:
  BulkData actBulk_;

};

}
}
#endif
