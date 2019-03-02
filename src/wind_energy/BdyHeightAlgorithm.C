/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "wind_energy/BdyHeightAlgorithm.h"
#include "Realm.h"
#include "utils/LinearInterpolation.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"

#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstdint>

namespace sierra {
namespace nalu {

RectilinearMeshHeightAlg::RectilinearMeshHeightAlg(
  Realm& realm,
  const YAML::Node& node
) : BdyHeightAlgorithm(realm)
{
  load(node);
}

void
RectilinearMeshHeightAlg::load(const YAML::Node& node)
{
  get_if_present(node, "wall_normal_direction", wallNormIndex_, wallNormIndex_);
  get_if_present(node, "minimum_height", hMin_, hMin_);
  get_if_present(node, "height_multiplier", heightMultiplier_, heightMultiplier_);
}

void
RectilinearMeshHeightAlg::calc_height_levels(
  stk::mesh::Selector& nodeSel,
  ScalarIntFieldType& indexField,
  std::vector<double>& gHeights)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();

  const auto bkts = bulk.get_buckets(stk::topology::NODE_RANK, nodeSel);
  const VectorFieldType* coords = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // Index of the wall normal coordinate
  const int iz = wallNormIndex_ - 1;

  // Determine unique height levels local to this proc
  std::unordered_set<uint64_t> hLevels;
  for (auto b: bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      const double* crd = stk::mesh::field_data(*coords, node);
      const double ht = crd[iz] - hMin_;

      const int htInt = static_cast<int>(std::round(ht * heightMultiplier_));
      hLevels.insert(htInt);
    }
  }

  // Gather All height levels on all MPI ranks
  const int nprocs = bulk.parallel_size();
  const int numLevelsLocal = hLevels.size();
  // Convert set to vector for MPI Send/Recv
  std::vector<uint64_t> hLevelsVec(numLevelsLocal);

  std::copy(hLevels.begin(), hLevels.end(), hLevelsVec.begin());

  // Communicate the number of levels across all MPI Ranks
  std::vector<int> lvlsPerProc(nprocs);
  MPI_Allgather(&numLevelsLocal, 1, MPI_INT, lvlsPerProc.data(), 1,
                MPI_INT, bulk.parallel());

  // Estimate total levels that must be gathered
  int nTotalLvls = std::accumulate(lvlsPerProc.begin(), lvlsPerProc.end(), 0);
  std::vector<int> offsets(nprocs+1, 0);
  std::vector<uint64_t> allLevels(nTotalLvls);

  offsets[0] = 0;
  for (int i=1; i <= nprocs; i++)
    offsets[i] = offsets[i-1] + lvlsPerProc[i-1];

  // All-to-all gather so that everyone knows what levels exist globally
  MPI_Allgatherv(hLevelsVec.data(), numLevelsLocal, MPI_UINT64_T,
                 allLevels.data(), lvlsPerProc.data(), offsets.data(),
                 MPI_UINT64_T, bulk.parallel());

  // Find the unique height levels across all MPI ranks
  std::unordered_set<uint64_t> gLevels;
  for (auto ht: allLevels) gLevels.insert(ht);

  // Sort height levels in ascending order
  const size_t nHeights = gLevels.size();
  std::vector<uint64_t> gLevelsVec(nHeights);
  std::copy(gLevels.begin(), gLevels.end(), gLevelsVec.begin());
  std::sort(gLevelsVec.begin(), gLevelsVec.end());

  // Nudge the upper bound so that indexing is captured correctly
  std::vector<double> heightVec(nHeights+1);
  for (size_t i=0; i < nHeights; i++)
    heightVec[i] = hMin_ + (static_cast<double>(gLevelsVec[i]) / heightMultiplier_);
  heightVec[nHeights] = heightVec[nHeights-1] + 1.0;

  // Perturb the z-coordinate value when comparing against available height levels
  const double eps = 1.0e-6;
  // Populate indexing so that all averaging can use this index for lookup
  for (auto b: bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      const double* crd = stk::mesh::field_data(*coords, node);
      int* hIdx = stk::mesh::field_data(indexField, node);
      const double ht = crd[iz] + eps;

      auto idx = utils::find_index(heightVec, ht);
      hIdx[0] = idx.second;
    }
  }

  // Populate sorted height data for statistics utilities
  gHeights.resize(nHeights);
  for (size_t i=0; i < nHeights; i++) {
    gHeights[i] = heightVec[i];
  }
}

}  // nalu
}  // sierra
