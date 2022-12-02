#include "mesh_motion/MotionWavesKernel.h"

#include <FieldTypeDef.h>
#include <NaluParsing.h>

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

namespace sierra {
namespace nalu {

MotionWavesKernel::MotionWavesKernel(
  stk::mesh::MetaData& meta, const YAML::Node& node)
  : NgpMotionKernel<MotionWavesKernel>()
{
  load(node);

  // declare divergence of mesh velocity for this motion
  isDeforming_ = true;
  ScalarFieldType* divV = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_mesh_velocity"));
  stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
}

void
MotionWavesKernel::load(const YAML::Node& node)
{
  // Get type of input prescribed wave
  std::string waveString = "Airy";
  get_if_present(node, "wave_model", waveString, waveString);
  // Get vertical mesh damping amplitude
  get_if_present(
    node, "mesh_damping_length", meshdampinglength_, meshdampinglength_);
  get_if_present(
    node, "mesh_damping_coeff", meshdampingcoeff_, meshdampingcoeff_);

  if (waveString == "Airy") {
    waveModel_ = 1;
    get_if_present(node, "wave_height", height_, height_);
    get_if_present(node, "wave_length", length_, length_);
    get_if_present(node, "water_depth", waterdepth_, waterdepth_);
    // Compute wave number
    k_ = 2 * M_PI / length_;
    // This calculates the wave frequency based on the linear dispersion
    // relationship
    omega_ = stk::math::pow(k_ * g_ * stk::math::tanh(k_ * waterdepth_), 0.5);
    c_ = omega_ / k_;
    // Over-write wave phase velocity if specified
    get_if_present(node, "phase_velocity", c_, c_);
    omega_ = c_ * k_;
    period_ = length_ / c_;
  } else if (waveString == "Stokes") {
    waveModel_ = 2;
    get_if_present(node, "Stokes_order", StokesOrder_, StokesOrder_);
    get_if_present(node, "wave_height", height_, height_);
    get_if_present(node, "wave_length", length_, length_);
    get_if_present(node, "water_depth", waterdepth_, waterdepth_);
    // Compute parameters
    k_ = 2. * M_PI / length_;
    Stokes_coefficients();
    Stokes_parameters();
    get_if_present(node, "phase_velocity", c_, c_);
  } else if (waveString == "Idealized") {
    waveModel_ = 3;
    get_if_present(node, "wave_height", height_, height_);
    get_if_present(node, "wave_length", length_, length_);
    get_if_present(node, "phase_velocity", c_, c_);
    k_ = 2 * M_PI / length_;
    omega_ = c_ * k_;
    period_ = length_ / c_;
  } else if (waveString == "Piston") {
    waveModel_ = 4;
    get_if_present(node, "wave_height", height_, height_);
    get_if_present(node, "frequency", omega_, omega_);
    get_if_present(node, "phase_velocity", c_, c_);
  } else {
    throw std::runtime_error("invalid wave_motion model specified ");
  }

  // Time parameters
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;
  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;
  get_if_present(node, "sea_level_z", sealevelz_, sealevelz_);

  get_if_present(node, "waves_rampup", do_rampup_, do_rampup_);
  if (do_rampup_) {
    get_if_present(
      node, "rampup_start_time", rampup_start_time_, rampup_start_time_);
    get_if_present(node, "rampup_period", rampup_period_, rampup_period_);
  }
} // namespace nalu

KOKKOS_FUNCTION
mm::TransMatType
MotionWavesKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& xyz)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  mm::ThreeDVecType disp;
  double phase = k_ * xyz[0] - omega_ * motionTime;
  if (waveModel_ == 1) {
    disp[0] = 0.;
    disp[1] = 0.;
    disp[2] =
      sealevelz_ +
      height_ / 2. * stk::math::cos(phase) *
        stk::math::pow(1 - xyz[2] / meshdampinglength_, meshdampingcoeff_);
  } else if (waveModel_ == 2) {
    disp[0] = 0.;
    disp[1] = 0.;
    disp[2] =
      sealevelz_ +
      (eps_ * stk::math::cos(phase) // first order term
       + stk::math::pow(eps_, 2) * b22_ *
           stk::math::cos(2. * phase) // second order term
       + stk::math::pow(eps_, 3) * b31_ *
           (stk::math::cos(phase) - stk::math::cos(3. * phase)) +
       stk::math::pow(eps_, 4) * b42_ *
         (stk::math::cos(2. * phase) + b44_ * stk::math::cos(4 * phase)) +
       stk::math::pow(eps_, 5) * (-(b53_ + b55_) * stk::math::cos(phase) +
                                  b53_ * stk::math::cos(3 * phase) +
                                  b55_ * stk::math::cos(5 * phase))) /
        k_ * stk::math::pow(1 - xyz[2] / meshdampinglength_, meshdampingcoeff_);
  } else if (waveModel_ == 3) {
    disp[0] = 0.;
    disp[1] = 0.;
    disp[2] =
      height_ / 2. * stk::math::sin(phase) *
      stk::math::pow(1 - xyz[2] / meshdampinglength_, meshdampingcoeff_);
  } else if (waveModel_ == 4) {
    disp[0] = 0.;
    disp[1] = 0.;
    disp[2] = c_ * motionTime;
    // height_ * stk::math::sin(omega_ * motionTime) *
    // stk::math::pow(1 - xyz[2] / meshdampinglength_, meshdampingcoeff_);
  }

  double fac = 0.0;
  if (do_rampup_ && time >= startTime_ && time < startTime_ + rampup_period_) {
    fac = stk::math::tanh(2.0 * (time - startTime_) / rampup_period_);
  } else {
    fac = 1.0;
  }

  // Build matrix for translating object
  transMat[0 * mm::matSize + 3] = disp[0] * fac;
  transMat[1 * mm::matSize + 3] = disp[1] * fac;
  transMat[2 * mm::matSize + 3] = disp[2] * fac;
  return transMat;
} // namespace nalu

KOKKOS_FUNCTION
mm::ThreeDVecType
MotionWavesKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& /* compTrans */,
  const mm::ThreeDVecType& mxyz,
  const mm::ThreeDVecType& /* cxyz */)
{
  mm::ThreeDVecType vel;

  if ((time < startTime_) || (time > endTime_))
    return vel;

  double motionTime = (time < endTime_) ? time : endTime_;

  double StreamwiseWaveVelocity = 0;
  double LateralWaveVelocity = 0;
  double VerticalWaveVelocity = 0;
  double phase = k_ * mxyz[0] - omega_ * motionTime;

  if (waveModel_ == 1) {
    StreamwiseWaveVelocity =
      omega_ * height_ / 2. * stk::math::cosh(k_ * waterdepth_) /
      stk::math::sinh(k_ * waterdepth_) * stk::math::cos(phase);
    VerticalWaveVelocity = omega_ * height_ / 2. * stk::math::sin(phase);
  } else if (waveModel_ == 2) {
    StreamwiseWaveVelocity =
      my_cosh_cos(1, 1, phase) + my_cosh_cos(2, 2, phase) +
      my_cosh_cos(3, 1, phase) + my_cosh_cos(3, 3, phase) +
      my_cosh_cos(4, 2, phase) + my_cosh_cos(4, 4, phase) +
      my_cosh_cos(5, 1, phase) + my_cosh_cos(5, 3, phase) +
      my_cosh_cos(5, 5, phase);
    VerticalWaveVelocity = my_sinh_sin(1, 1, phase) + my_sinh_sin(2, 2, phase) +
                           my_sinh_sin(3, 1, phase) + my_sinh_sin(3, 3, phase) +
                           my_sinh_sin(4, 2, phase) + my_sinh_sin(4, 4, phase) +
                           my_sinh_sin(5, 1, phase) + my_sinh_sin(5, 3, phase) +
                           my_sinh_sin(5, 5, phase);
    StreamwiseWaveVelocity *= c0_ * stk::math::sqrt(g_ / stk::math::pow(k_, 3));
    VerticalWaveVelocity *= c0_ * stk::math::sqrt(g_ / stk::math::pow(k_, 3));
  } else if (waveModel_ == 3) {
    StreamwiseWaveVelocity = omega_ * height_ / 2. * stk::math::sin(phase);
    VerticalWaveVelocity = -omega_ * height_ / 2. * stk::math::cos(phase);
  } else if (waveModel_ == 4) {
    StreamwiseWaveVelocity = 0.;
    VerticalWaveVelocity = c_;
  }

  double fac = 0.0;
  if (do_rampup_ && time >= startTime_ && time < startTime_ + rampup_period_) {
    fac = stk::math::tanh(2.0 * (time - startTime_) / rampup_period_);
  } else {
    fac = 1.0;
  }

  if (mxyz[2] < sealevelz_ + DBL_EPSILON) {
    vel[0] = StreamwiseWaveVelocity * fac;
    vel[1] = LateralWaveVelocity * fac;
    vel[2] = VerticalWaveVelocity * fac;
  } else {
    vel[0] = 0.;
    vel[1] = 0.;
    vel[2] = 0.;
  }
  return vel;
}

/* Define the Stokes expansion coefficients based on "A Fifth-Order Stokes
 * Theory for Steady Waves" (J. D. Fenton, 1985)
 */
void
MotionWavesKernel::Stokes_coefficients()
{
  double kd = k_ * waterdepth_;
  if (kd > 50 * M_PI)
    kd = 50 * M_PI; // Limited value

  double S = 2 * stk::math::exp(2 * kd) / (stk::math::exp(4 * kd) + 1);
  double Sh = stk::math::sinh(kd);
  double Th = stk::math::tanh(kd);
  double CTh = (1 + stk::math::exp(-2. * kd)) /
               (1 - stk::math::exp(-2 * kd)); // Hyperbolic cotangent

  a11_ = 1. / stk::math::sinh(kd); // Hyperbolic cosecant
  c0_ = stk::math::sqrt(Th);
  // Second order coefficients
  a22_ = 3. * stk::math::pow(S, 2) / (2 * stk::math::pow(1 - S, 2));
  b22_ = CTh * (1 + 2. * S) / (2 * (1 - S));
  c2_ = stk::math::sqrt(Th) * (2 + 7 * stk::math::pow(S, 2)) /
        (4 * stk::math::pow(1 - S, 2));
  d2_ = -stk::math::sqrt(CTh) / 2.;
  e2_ = Th * (2 + 2 * S + 5 * stk::math::pow(S, 2)) /
        (4 * stk::math::pow(1 - S, 2));
  if (StokesOrder_ == 2)
    return;

  // Third order coefficients
  a31_ = (-4 - 20 * S + 10 * stk::math::pow(S, 2) - 13 * stk::math::pow(S, 3)) /
         (8 * Sh * stk::math::pow(1 - S, 3));
  a33_ = (-2 * stk::math::pow(S, 2) + 11 * stk::math::pow(S, 3)) /
         (8 * Sh * stk::math::pow(1 - S, 3));
  b31_ = -3 *
         (1 + 3 * S + 3 * stk::math::pow(S, 2) + 2 * stk::math::pow(S, 3)) /
         (8 * stk::math::pow(1 - S, 3));
  if (StokesOrder_ == 3)
    return;

  // Fourth order coefficients
  a42_ = (12 * S - 14 * stk::math::pow(S, 2) - 264 * stk::math::pow(S, 3) -
          45 * stk::math::pow(S, 4) - 13 * stk::math::pow(S, 5)) /
         (24 * stk::math::pow(1 - S, 5));
  a44_ = (10 * stk::math::pow(S, 3) - 174 * stk::math::pow(S, 4) +
          291 * stk::math::pow(S, 5) + 278 * stk::math::pow(S, 6)) /
         (48 * (3 + 2 * S) * stk::math::pow(1 - S, 5));
  b42_ = CTh *
         (6 - 26 * S - 182 * stk::math::pow(S, 2) - 204 * stk::math::pow(S, 3) -
          25 * stk::math::pow(S, 4) + 26 * stk::math::pow(S, 5)) /
         (6 * (3 + 2 * S) * stk::math::pow(1 - S, 4));
  b44_ = CTh *
         (24 + 92 * S + 122 * stk::math::pow(S, 2) + 66 * stk::math::pow(S, 3) +
          67 * stk::math::pow(S, 4) + 34 * stk::math::pow(S, 5)) /
         (24 * (3 + 2 * S) * stk::math::pow(1 - S, 4));
  c4_ = stk::math::sqrt(Th) *
        (4 + 32 * S - 116 * stk::math::pow(S, 2) - 400 * stk::math::pow(S, 3) -
         71 * stk::math::pow(S, 4) + 146 * stk::math::pow(S, 5)) /
        (32 * stk::math::pow(1 - S, 5));
  d4_ = stk::math::sqrt(CTh) *
        (2 + 4 * S + stk::math::pow(S, 2) + 2 * stk::math::pow(S, 3)) /
        (8 * stk::math::pow(1 - S, 3));
  e4_ = Th *
        (8 + 12 * S - 152 * stk::math::pow(S, 2) - 308 * stk::math::pow(S, 3) -
         42 * stk::math::pow(S, 4) + 77 * stk::math::pow(S, 5)) /
        (32 * stk::math::pow(1 - S, 5));
  if (StokesOrder_ == 4)
    return;

  // Fifth order coefficients
  a51_ = (-1184 + 32 * S + 13232 * stk::math::pow(S, 2) +
          21712 * stk::math::pow(S, 3) + 20940 * stk::math::pow(S, 4) +
          12554 * stk::math::pow(S, 5) - 500 * stk::math::pow(S, 6) -
          3341 * stk::math::pow(S, 7) - 670 * stk::math::pow(S, 8)) /
         (64 * Sh * (3 + 2 * S) * (4 + S) * stk::math::pow(1 - S, 6));
  a53_ = (4 * S + 105 * pow(S, 2) + 198 * stk::math::pow(S, 3) -
          1376 * stk::math::pow(S, 4) - 1302 * stk::math::pow(S, 5) -
          117 * stk::math::pow(S, 6) + 58 * stk::math::pow(S, 7)) /
         (32 * Sh * (3 + 2 * S) * stk::math::pow(1 - S, 6));
  a55_ = (-6 * stk::math::pow(S, 3) + 272 * stk::math::pow(S, 4) -
          1552 * stk::math::pow(S, 5) + 852 * stk::math::pow(S, 6) +
          2029 * stk::math::pow(S, 7) + 430 * stk::math::pow(S, 8)) /
         (64 * Sh * (3 + 2 * S) * (4 + S) * stk::math::pow(1 - S, 6));
  b53_ = 9 *
         (132 + 17 * S - 2216 * stk::math::pow(S, 2) -
          5897 * stk::math::pow(S, 3) - 6292 * stk::math::pow(S, 4) -
          2687 * stk::math::pow(S, 5) + 194 * stk::math::pow(S, 6) +
          467 * stk::math::pow(S, 7) + 82 * stk::math::pow(S, 8)) /
         (128 * (3 + 2 * S) * (4 + S) * stk::math::pow(1 - S, 6));
  b55_ = 5 *
         (300 + 1579 * S + 3176 * stk::math::pow(S, 2) +
          2949 * stk::math::pow(S, 3) + 1188 * stk::math::pow(S, 4) +
          675 * stk::math::pow(S, 5) + 1326 * stk::math::pow(S, 6) +
          827 * stk::math::pow(S, 7) + 130 * stk::math::pow(S, 8)) /
         (384 * (3 + 2 * S) * (4 + S) * stk::math::pow(1 - S, 6));
  if (StokesOrder_ == 5)
    return;

  if (StokesOrder_ > 5 || StokesOrder_ < 2) {
    throw std::runtime_error(
      "invalid stokes order speficied. It should be between 2,3,4 or 5 ");
  }
}

void
MotionWavesKernel::Stokes_parameters()
{
  k_ = 2 * M_PI / length_;
  eps_ = k_ * height_ / 2.; // Steepness (ka)
  c_ = (c0_ + stk::math::pow(eps_, 2) * c2_ + stk::math::pow(eps_, 4) * c4_) *
       stk::math::sqrt(g_ / k_);
  Q_ =
    c_ * waterdepth_ * stk::math::sqrt(stk::math::pow(k_, 3) / g_) +
    d2_ * stk::math::pow(eps_, 2) +
    d4_ * stk::math::pow(eps_, 4) * stk::math::sqrt(g_ / stk::math::pow(k_, 3));
  cs_ = c_ - Q_;
  period_ = length_ / c_;
  omega_ = c_ * k_;
  return;
}

KOKKOS_FUNCTION
double
MotionWavesKernel::my_cosh_cos(int i, int j, const double& phase)
{
  double D = 0.0;
  if (i == 1 && j == 1)
    D = a11_;
  if (i == 2 && j == 2)
    D = a22_;
  if (i == 3 && j == 1)
    D = a31_;
  if (i == 3 && j == 3)
    D = a33_;
  if (i == 4 && j == 2)
    D = a42_;
  if (i == 4 && j == 4)
    D = a44_;
  if (i == 5 && j == 1)
    D = a51_;
  if (i == 5 && j == 3)
    D = a53_;
  if (i == 5 && j == 5)
    D = a55_;

  return stk::math::pow(eps_, i) * D * j * k_ *
         stk::math::cosh(j * k_ * waterdepth_) * stk::math::cos(j * phase);
}

KOKKOS_FUNCTION
double
MotionWavesKernel::my_sinh_sin(int i, int j, const double& phase)
{
  double D = 0.0;
  if (i == 1 && j == 1)
    D = a11_;
  if (i == 2 && j == 2)
    D = a22_;
  if (i == 3 && j == 1)
    D = a31_;
  if (i == 3 && j == 3)
    D = a33_;
  if (i == 4 && j == 2)
    D = a42_;
  if (i == 4 && j == 4)
    D = a44_;
  if (i == 5 && j == 1)
    D = a51_;
  if (i == 5 && j == 3)
    D = a53_;
  if (i == 5 && j == 5)
    D = a55_;

  return stk::math::pow(eps_, i) * D * j * k_ *
         stk::math::sinh(j * k_ * waterdepth_) * stk::math::sin(j * phase);
}

void
MotionWavesKernel::get_StokesCoeff(StokesCoeff* stokes)
{
  stokes->k = k_;
  stokes->d = height_;
  stokes->a11 = a11_;
  stokes->a22 = a22_;
  stokes->a31 = a31_;
  stokes->a33 = a33_;
  stokes->a42 = a42_;
  stokes->a44 = a44_;
  stokes->a51 = a51_;
  stokes->a53 = a53_;
  stokes->a55 = a55_;
  stokes->b22 = b22_;
  stokes->b31 = b31_;
  stokes->b42 = b42_;
  stokes->b44 = b44_;
  stokes->b53 = b53_;
  stokes->b55 = b55_;
  stokes->c0 = c0_;
  stokes->c2 = c2_;
  stokes->c4 = c4_;
  stokes->d2 = d2_;
  stokes->d4 = d4_;
  stokes->e2 = e2_;
  stokes->e4 = e4_;
}

} // namespace nalu
} // namespace sierra
