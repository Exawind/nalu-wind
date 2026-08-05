#ifndef KYNEMAFMBBASE_H
#define KYNEMAFMBBASE_H

#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace kynema_ugf {

class KynemaFMBBase
{
public:
  virtual ~KynemaFMBBase() = default;

  virtual void
  setup(double dtKynemaUGF, std::shared_ptr<stk::mesh::BulkData> bulk) = 0;

  virtual void initialize(int restartFreqKynemaUGF, double curTime) = 0;

  virtual void map_displacements(double currentTime, bool updateCurCoor) = 0;

  virtual void
  advance_struct_timestep(const double currentTime, const double dT) = 0;

  virtual void map_loads(const double currentTime) = 0;

  virtual stk::mesh::PartVector get_mesh_blocks() const = 0;
};

} // namespace kynema_ugf
} // namespace sierra

#endif // KYNEMAFMBBASE_H