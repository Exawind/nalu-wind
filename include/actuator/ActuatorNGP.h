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

namespace sierra{
namespace nalu{

// Kokkos compatible functors that can be implemented based on
// BulkData Type

template<typename BulkData>
struct ActuatorPreIteration{
  BulkData& bulk_;
  ActuatorPreIteration(BulkData& bulk):bulk_(bulk){}
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
  ActuatorPreIteration<BulkData> preIteration_;
  ActuatorComputePointLocation<BulkData> computePointLocation_;
  ActuatorInterpolateFieldValues<BulkData> interpolateFieldValues_;
  ActuatorSpreadForces<BulkData> spreadForces_;
  ActuatorPostIteration<BulkData> postIteration_;

};

}
}
#endif
