#ifndef KYNEMAFMBBASE_H
#define KYNEMAFMBBASE_H

#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Part.hpp>
#include "FieldTypeDef.h"
#include "aero/fsi/CalcLoads.h"

namespace sierra {
namespace kynema_ugf {

struct PointData
{
  std::array<double, 7> pos;
  std::array<double, 6> vel;
  std::array<double, 6> loads;
  std::array<double, 3> center_of_mass;
};

struct PointMass
{
  std::vector<std::string> forcing_surface_names;
  std::vector<std::string> moving_mesh_block_names;
  stk::mesh::PartVector forcing_surfaces;
  stk::mesh::PartVector moving_mesh_blocks;
  GenericFieldType* total_force = nullptr;
  std::shared_ptr<stk::mesh::BulkData> bulk = nullptr;
  std::shared_ptr<CalcLoads> calc_loads = nullptr;
  PointData p_data;
};

class KynemaFMBBase
{
public:
  virtual ~KynemaFMBBase() = default;

  virtual void
  setup(double dtKynemaUGF, std::shared_ptr<stk::mesh::BulkData> bulk) = 0;

  virtual void initialize(int restartFreqKynemaUGF, double curTime) = 0;

  virtual void map_displacements(double currentTime, bool updateCurCoor) = 0;

  void map_displacements_point(PointMass& point, bool updateCur);

  virtual void
  advance_struct_timestep(const double currentTime, const double dT) = 0;

  virtual void map_loads(const double currentTime) = 0;

  void map_loads_point(PointMass& point);

  virtual stk::mesh::PartVector get_mesh_blocks() const = 0;

  std::vector<PointMass> point_bodies_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif // KYNEMAFMBBASE_H