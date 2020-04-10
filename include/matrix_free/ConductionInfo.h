#ifndef CONDUCTION_INFO_H
#define CONDUCTION_INFO_H

namespace sierra {
namespace nalu {
namespace matrix_free {

struct conduction_info
{
  enum {
    TEMPERATURE_NP1 = 0,
    TEMPERATURE_NP0 = 1,
    TEMPERATURE_NM1 = 2,
    ALPHA = 3,
    LAMBDA = 4
  };

  static constexpr int num_physics_fields = 5;
  static constexpr int num_coefficient_fields = 2;
  static constexpr auto coord_name = "coordinates";
  static constexpr auto q_name = "temperature";
  static constexpr auto qtmp_name = "tTmp";
  static constexpr auto volume_weight_name = "volumetric_heat_capacity";
  static constexpr auto diffusion_weight_name = "thermal_conductivity";
  static constexpr auto gid_name = "tpet_global_id";
  static constexpr auto qbc_name = "temperature_bc";
  static constexpr auto flux_name = "heat_flux_bc";
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
