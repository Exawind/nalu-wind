/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "UnitTestFieldUtils.h"

namespace unit_test_utils {

double field_norm(const ScalarFieldType& field, const stk::mesh::BulkData& bulk, stk::mesh::Selector selector)
{
  stk::ParallelMachine comm = bulk.parallel();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);

  size_t N = 0;
  size_t g_N = 0;
  double norm = 0.0;
  double g_norm = 0.0;

  Kokkos::parallel_for(sierra::nalu::HostTeamPolicy(buckets.size(), Kokkos::AUTO), NONCONST_LAMBDA(const sierra::nalu::TeamHandleType& team)
  {
      const stk::mesh::Bucket& bkt = *buckets[team.league_rank()];
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bkt.size()), NONCONST_LAMBDA(const size_t& j)
      {
          double node_value = *stk::mesh::field_data(field, bkt[j]);
          Kokkos::atomic_add(&N, (size_t)1);
          Kokkos::atomic_add(&norm, (node_value * node_value));
      });
  });

  stk::all_reduce_sum(comm, &N, &g_N, 1);
  stk::all_reduce_sum(comm, &norm, &g_norm, 1);
  g_norm = std::sqrt(g_norm/g_N);

  return g_norm;
}

}
