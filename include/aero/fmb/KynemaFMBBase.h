#ifndef KYNEMAFMBBASE_H
#define KYNEMAFMBBASE_H

#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Part.hpp>
#include "FieldTypeDef.h"
#include "aero/fsi/CalcLoads.h"
#include "aero/fsi/CalcLoadsAssembled.h"

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

template <typename Space>
struct BeamDataT
{
  using scalar_view_type = Kokkos::View<double*, Space>;
  using pos_view_type = Kokkos::View<double* [7], Space>;
  using vel_view_type = Kokkos::View<double* [6], Space>;
  using load_view_type = Kokkos::View<double* [6], Space>;

  scalar_view_type bary_weights; // Barycentric weights for interpolation nodes
  scalar_view_type node_xi; // Non-dimensional location of each node along the beam
  pos_view_type pos;       // First 3 are position, last 4 are quaternion orientation
  pos_view_type disp;      // First 3 are translational displacement, last 4 are quaternion rotations
  vel_view_type vel;       // First 3 are translational velocity, last 3 are rotational velocity
  load_view_type loads;     // First 3 are forces, last 3 are moments

  BeamDataT() = default;

  explicit BeamDataT(const std::size_t nNodes, const std::string& labelPrefix = "beam")
  {
    resize(nNodes, labelPrefix);
  }

  void resize(const std::size_t nNodes, const std::string& labelPrefix = "beam")
  {
    bary_weights = scalar_view_type(labelPrefix + "_bary_weights", nNodes);
    node_xi = scalar_view_type(labelPrefix + "_node_xi", nNodes);
    pos = pos_view_type(labelPrefix + "_pos", nNodes);
    disp = pos_view_type(labelPrefix + "_disp", nNodes);
    vel = vel_view_type(labelPrefix + "_vel", nNodes);
    loads = load_view_type(labelPrefix + "_loads", nNodes);
  }

  std::size_t size() const { return node_xi.extent(0); }

};

using BeamDataDevice = BeamDataT<MemSpace>;
using BeamDataHost = BeamDataT<HostSpace>;
using BeamData = BeamDataDevice;

struct BeamBody
{

  size_t n_nodes = 0;

  BeamData beam_data;
  BeamDataHost beam_data_host;

  std::vector<std::string> forcing_surface_names;
  std::vector<std::string> moving_mesh_block_names;
  stk::mesh::PartVector forcing_surfaces;
  stk::mesh::PartVector moving_mesh_blocks;

  GenericFieldType* total_force = nullptr;
  std::shared_ptr<stk::mesh::BulkData> bulk = nullptr;
  std::shared_ptr<CalcLoadsAssembled> calc_loads = nullptr;

};


class KynemaFMBBase
{
public:
  virtual ~KynemaFMBBase() = default;

  virtual void
  setup(double dtKynemaUGF, std::shared_ptr<stk::mesh::BulkData> bulk) = 0;

  virtual void initialize(int restartFreqKynemaUGF, double curTime) = 0;

  virtual void map_displacements(double currentTime, bool updateCurCoor) = 0;

  void compute_mapping_beam(BeamBody& beam);

  void map_displacements_beam(BeamBody& beam, bool updateCur);

  void map_displacements_point(PointMass& point, bool updateCur);

  virtual void
  advance_struct_timestep(const double currentTime, const double dT) = 0;

  virtual void map_loads(const double currentTime) = 0;

  void map_loads_beam(BeamBody& beam);

  void map_loads_point(PointMass& point);

  virtual stk::mesh::PartVector get_mesh_blocks() const = 0;

  std::vector<PointMass> point_bodies_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif // KYNEMAFMBBASE_H