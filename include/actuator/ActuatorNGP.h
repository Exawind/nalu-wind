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

#include <Kokkos_Core.hpp>
namespace sierra{
namespace nalu{

// Kokkos compatible functors that can be implemented based on
// BulkData Type

template<typename BulkData>
struct ActuatorPreIteration{
  BulkData& bulk_;
  ActuatorPreIteration(BulkData& bulk):bulk_(bulk){}
  void operator()(const std::size_t index);
};

template<typename BulkData>
struct ActuatorComputePointLocation{
  BulkData& bulk_;
  ActuatorComputePointLocation(BulkData& bulk):bulk_(bulk){}
  void operator()(const std::size_t index);
};

template<typename BulkData>
struct ActuatorInterpolateFieldValues{
  BulkData& bulk_;
  ActuatorInterpolateFieldValues(BulkData& bulk):bulk_(bulk){}
  void operator()(const std::size_t index);
};

template<typename BulkData>
struct ActuatorSpreadForces{
  BulkData& bulk_;
  ActuatorSpreadForces(BulkData& bulk):bulk_(bulk){}
  void operator()(const std::size_t index);
};

template<typename BulkData>
struct ActuatorPostIteration{
  BulkData& bulk_;
  ActuatorPostIteration(BulkData& bulk):bulk_(bulk){}
  void operator()(const std::size_t index);
};


template<typename MetaData, typename BulkData>
class Actuator
{
public:
  Actuator(MetaData actMeta);
  void execute()
  {
    //TODO(psakiev) set execution space i.e. range policy
    const std::size_t nP = actBulk_.total_num_points();
    Kokkos::parallel_for(nP, preIteration_);
    Kokkos::parallel_for(nP, computePointLocation_);
    Kokkos::parallel_for(nP, interpolateFieldValues_);
    Kokkos::parallel_for(nP, spreadForces_);
    Kokkos::parallel_for(nP, postIteration_);
  }
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
