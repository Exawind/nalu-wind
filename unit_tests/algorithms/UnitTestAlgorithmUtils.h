/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTALGORITHMUTILS_H
#define UNITTESTALGORITHMUTILS_H

#include <gtest/gtest.h>
#include "UnitTestUtils.h"

#include "Algorithm.h"
#include "SupplementalAlgorithm.h"
#include "UnitTestKokkosUtils.h"

#ifndef KOKKOS_ENABLE_CUDA

namespace unit_test_algorithm_utils {

/** Driver class that mimics Assemble*SolverAlgorithm
 *
 * It is the caller's responsibility to populate `activeSuppAlgs_` and call the
 * `setup` method on the activated SuppAlg before calling the `execute` method
 * of this class.
 *
 */
class TestSupplementalAlgorithmDriver
{
public:
  TestSupplementalAlgorithmDriver(
    const stk::mesh::BulkData& bulk)
    : bulk_(bulk)
  {}

  void nodal_execute()
  {
    const stk::mesh::MetaData& meta = bulk_.mesh_meta_data();

    const stk::mesh::Selector selector = meta.locally_owned_part();

    const auto& buckets = bulk_.get_buckets(stk::topology::NODE_RANK,
                                            selector);

    lhs_norm_ = 0.0;
    rhs_norm_ = 0.0;
    N_ = 0;

    kokkos_thread_team_bucket_loop(buckets, [&](stk::mesh::Entity node) {
	for (size_t i=0; i < activeSuppAlgs_.size(); ++i){
	  double lhs_value = 0.0;
	  double rhs_value = 0.0;

	  activeSuppAlgs_[i]->node_execute(&lhs_value, &rhs_value, node);

	  Kokkos::atomic_add(&lhs_norm_, (lhs_value * lhs_value));
	  Kokkos::atomic_add(&rhs_norm_, (rhs_value * rhs_value));
	}

	Kokkos::atomic_add(&N_, (size_t)1);
      });
  }

  inline double get_lhs_norm() {return unit_test_utils::global_norm(lhs_norm_, N_, bulk_.parallel());}
  inline double get_rhs_norm() {return unit_test_utils::global_norm(rhs_norm_, N_, bulk_.parallel());}

  std::vector<sierra::nalu::SupplementalAlgorithm*> activeSuppAlgs_;

private:
  const stk::mesh::BulkData& bulk_;
  double lhs_norm_;
  double rhs_norm_;
  size_t N_;
};

}

#endif /* KOKKOS_ENABLE_CUDA */

#endif /* UNITTESTALGORITHMUTILS_H */
