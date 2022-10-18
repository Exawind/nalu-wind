#ifndef MOTIONWAVES_H
#define MOTIONWAVES_H

#include "NgpMotion.h"

#include <string>

namespace stk {
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {
class MotionWavesKernel : public NgpMotionKernel<MotionWavesKernel>
{
public:
  MotionWavesKernel(stk::mesh::MetaData&, const YAML::Node&);

  MotionWavesKernel() = default;

  virtual ~MotionWavesKernel() = default;

  /** Function to compute motion-specific transformation matrix
   *
   * @param[in] time Current time
   * @param[in] xyz  Coordinates
   * @return Transformation matrix
   */
  KOKKOS_FUNCTION
  virtual mm::TransMatType
  build_transformation(const double& time, const mm::ThreeDVecType& xyz);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time      Current time
   * @param[in]  compTrans Transformation matrix
   *                       including all motions
   * @param[in]  mxyz      Model coordinates
   * @param[in]  cxyz      Transformed coordinates
   * @return Velocity vector associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual mm::ThreeDVecType compute_velocity(
    const double& time,
    const mm::TransMatType& compTrans,
    const mm::ThreeDVecType& mxyz,
    const mm::ThreeDVecType& cxyz);

  struct StokesCoeff
  {
    double k;
    double d;
    double a11;
    double a22;
    double a31;
    double a33;
    double a42;
    double a44;
    double a51;
    double a53;
    double a55;
    double b22;
    double b31;
    double b42;
    double b44;
    double b53;
    double b55;
    double c0;
    double c2;
    double c4;
    double d2;
    double d4;
    double e2;
    double e4;
  };

  void get_StokesCoeff(StokesCoeff* stokes);

private:
  void load(const YAML::Node&);

  void Stokes_coefficients();
  void Stokes_parameters();

  KOKKOS_FUNCTION
  double my_sinh_sin(int i, int j, const double& phase);

  KOKKOS_FUNCTION
  double my_cosh_cos(int i, int j, const double& phase);

  const double g_{9.81};

  int waveModel_{1};

  // General parameters for waves
  double height_{0.1};     // Wave height
  double period_{1.0};     // Wave period
  double length_{1.0};     // Wave length
  double waterdepth_{100}; // Water depth
  double omega_{
    2. * M_PI}; // Angular frequency omega=2*pi/tau (tau being the period)
  double k_{
    2. *
    M_PI}; // Angular wavenumber k=2*pi/lambda (lambda being the wavenumber)
  double sealevelz_{0.0}; // Sea level assumed to be at z=0
  double c_{1.};          // wave phase velocity c

  bool do_rampup_{false};      // Logic to allow to ramp up waves
  double rampup_period_{10.0}; // rampup
  double rampup_start_time_{0.};

  // Stokes waves parameters
  int StokesOrder_{2}; // Stokes order - it defaults to 2
  double a11_{0.};
  double a22_{0.};
  double a31_{0.};
  double a33_{0.};
  double a42_{0.};
  double a44_{0.};
  double a51_{0.};
  double a53_{0.};
  double a55_{0.};
  double b22_{0.};
  double b31_{0.};
  double b42_{0.};
  double b44_{0.};
  double b53_{0.};
  double b55_{0.};
  double c0_{0.};
  double c2_{0.};
  double c4_{0.};
  double d2_{0.};
  double d4_{0.};
  double e2_{0.};
  double e4_{0.};
  double eps_{0.1};
  double Q_{0.};
  double cs_{0.2}; // Mean Stokes drift speed

  // Deformation damping function
  double meshdampinglength_{1000};
  int meshdampingcoeff_{3};
};

} // namespace nalu
} // namespace sierra

#endif /* MOTIONWAVES_H */
