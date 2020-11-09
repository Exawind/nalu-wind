
#include "mesh_motion/MotionWavesKernel.h"

#include <NaluParsing.h>
#include "utils/ComputeVectorDivergence.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra {
namespace nalu {

MotionWavesKernel::MotionWavesKernel(stk::mesh::MetaData& meta, const YAML::Node& node)
  : NgpMotionKernel<MotionWavesKernel>()
{
  load(node);

    // declare divergence of mesh velocity for this motion
    ScalarFieldType* divV = &(meta.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "div_mesh_velocity"));
    stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);

}

void
MotionWavesKernel::load(const YAML::Node& node)
{
  // Get type of input prescribed wave
  get_if_present(node, "wave_model", waveModel_, waveModel_);
  // Get vertical mesh damping amplitude
  get_if_present(
    node, "mesh_damping_length", meshdampinglength_, meshdampinglength_);
  get_if_present(
    node, "mesh_damping_coeff", meshdampingcoeff_, meshdampingcoeff_);

  if (waveModel_ == "Airy") {
    get_if_present(node, "wave_height", height_, height_);
    get_if_present(node, "wave_length", length_, length_);
    get_if_present(node, "water_depth", waterdepth_, waterdepth_);
    //Compute wave number
    k_=2*M_PI/length_;
    // This calculates the wave frequency based on the linear dispersion relationship
    omega_ = std::pow(k_ * g_ * std::tanh(k_ * waterdepth_), 0.5);
    c_ = omega_ / k_;
    // Over-write wave phase velocity if specified
    get_if_present(node, "phase_velocity", c_, c_);
    omega_ = c_ * k_;
    period_ = length_ / c_;
  } else if (waveModel_ == "Stokes") {
    get_if_present(node, "Stokes_order", StokesOrder_, StokesOrder_);
    get_if_present(node, "wave_height", height_, height_);
    get_if_present(node, "wave_length", length_, length_);
    get_if_present(node, "water_depth", waterdepth_, waterdepth_);
    // Compute parameters
    k_ = 2. * M_PI / length_;
    Stokes_coefficients();
    Stokes_parameters();
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
}

void
MotionWavesKernel::build_transformation(const double time, const double* xyz)
{
  if (time < (startTime_))
    return;

  double motionTime = (time < endTime_) ? time : endTime_;

  double phase = k_ * xyz[0] - omega_ * motionTime;
  ThreeDVecType curr_disp = {};
  if (waveModel_ == "Airy") {
    curr_disp[0] = 0.;
    curr_disp[1] = 0.;
    curr_disp[2] =
      sealevelz_ +
      height_ / 2. * std::cos(phase) *
        std::pow(1 - xyz[2] / meshdampinglength_, meshdampingcoeff_);
  } else if (waveModel_ == "Stokes") {
    curr_disp[0] = 0.;
    curr_disp[1] = 0.;
    curr_disp[2] =
      sealevelz_ +
      (eps_ * std::cos(phase)                            // first order term
       + std::pow(eps_, 2) * b22_ * std::cos(2. * phase) // second order term
       + std::pow(eps_, 3) * b31_ * (std::cos(phase) - std::cos(3. * phase)) +
       std::pow(eps_, 4) * b42_ *
         (std::cos(2. * phase) + b44_ * std::cos(4 * phase)) +
       std::pow(eps_, 5) *
         (-(b53_ + b55_) * std::cos(phase) + b53_ * std::cos(3 * phase) +
          b55_ * std::cos(5 * phase))) /
        k_ * std::pow(1 - xyz[2] / meshdampinglength_, meshdampingcoeff_);
  } else {
    throw std::runtime_error("invalid wave_motion model specified ");
  }
  translation_mat(curr_disp);
}

void
MotionWavesKernel::translation_mat(const ThreeDVecType& curr_disp)
{
  reset_mat(transMat_);

  // Build matrix for translating object
  transMat_[0][3] = curr_disp[0];
  transMat_[1][3] = curr_disp[1];
  transMat_[2][3] = curr_disp[2];
}

void MotionWavesKernel::compute_velocity(
  const double time,
  const TransMatType& /* compTrans */,
  const double* mxyz,
  const double* /* cxyz */,
  ThreeDVecType& vel)
{
  if((time < startTime_) || (time > endTime_)) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      vel[d] = 0.0;

    return;
  }

  double motionTime = (time < endTime_) ? time : endTime_;

  double StreamwiseWaveVelocity = 0;
  double LateralWaveVelocity = 0;
  double VerticalWaveVelocity = 0;
  double phase = k_ * mxyz[0] - omega_ * motionTime;

  if (waveModel_ == "Airy") {
    StreamwiseWaveVelocity = omega_ * height_ / 2. *
                             std::cosh(k_ * waterdepth_) /
                             std::sinh(k_ * waterdepth_) * std::cos(phase);
    VerticalWaveVelocity = omega_ * height_ / 2. * std::sin(phase);
  } else if (waveModel_ == "Stokes") {
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
    StreamwiseWaveVelocity *= c0_ * std::sqrt(g_ / std::pow(k_, 3));
    VerticalWaveVelocity *= c0_ * std::sqrt(g_ / std::pow(k_, 3));
  } else {
    throw std::runtime_error("invalid wave_motion model specified ");
  }

  if (mxyz[2] < sealevelz_ + DBL_EPSILON) {
    vel[0] = StreamwiseWaveVelocity;
    vel[1] = LateralWaveVelocity;
    vel[2] = VerticalWaveVelocity;
  } else {
    vel[0] = 0.;
    vel[1] = 0.;
    vel[2] = 0.;
  }
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

  double S = 2 * std::exp(2 * kd) / (std::exp(4 * kd) + 1);
  double Sh = std::sinh(kd);
  double Th = std::tanh(kd);
  double CTh =
    (1 + std::exp(-2. * kd)) / (1 - std::exp(-2 * kd)); // Hyperbolic cotangent

  a11_ = 1. / std::sinh(kd); // Hyperbolic cosecant
  c0_ = std::sqrt(Th);
  // Second order coefficients
  a22_ = 3. * std::pow(S, 2) / (2 * std::pow(1 - S, 2));
  b22_ = CTh * (1 + 2. * S) / (2 * (1 - S));
  c2_ = std::sqrt(Th) * (2 + 7 * std::pow(S, 2)) / (4 * std::pow(1 - S, 2));
  d2_ = -std::sqrt(CTh) / 2.;
  e2_ = Th * (2 + 2 * S + 5 * std::pow(S, 2)) / (4 * std::pow(1 - S, 2));
  if (StokesOrder_ == 2)
    return;

  // Third order coefficients
  a31_ = (-4 - 20 * S + 10 * std::pow(S, 2) - 13 * std::pow(S, 3)) /
         (8 * Sh * std::pow(1 - S, 3));
  a33_ =
    (-2 * std::pow(S, 2) + 11 * std::pow(S, 3)) / (8 * Sh * std::pow(1 - S, 3));
  b31_ = -3 * (1 + 3 * S + 3 * std::pow(S, 2) + 2 * std::pow(S, 3)) /
         (8 * std::pow(1 - S, 3));
  if (StokesOrder_ == 3)
    return;

  // Fourth order coefficients
  a42_ = (12 * S - 14 * std::pow(S, 2) - 264 * std::pow(S, 3) -
          45 * std::pow(S, 4) - 13 * std::pow(S, 5)) /
         (24 * std::pow(1 - S, 5));
  a44_ = (10 * std::pow(S, 3) - 174 * std::pow(S, 4) + 291 * std::pow(S, 5) +
          278 * std::pow(S, 6)) /
         (48 * (3 + 2 * S) * std::pow(1 - S, 5));
  b42_ = CTh *
         (6 - 26 * S - 182 * std::pow(S, 2) - 204 * std::pow(S, 3) -
          25 * std::pow(S, 4) + 26 * std::pow(S, 5)) /
         (6 * (3 + 2 * S) * std::pow(1 - S, 4));
  b44_ = CTh *
         (24 + 92 * S + 122 * std::pow(S, 2) + 66 * std::pow(S, 3) +
          67 * std::pow(S, 4) + 34 * std::pow(S, 5)) /
         (24 * (3 + 2 * S) * std::pow(1 - S, 4));
  c4_ = std::sqrt(Th) *
        (4 + 32 * S - 116 * std::pow(S, 2) - 400 * std::pow(S, 3) -
         71 * std::pow(S, 4) + 146 * std::pow(S, 5)) /
        (32 * std::pow(1 - S, 5));
  d4_ = std::sqrt(CTh) * (2 + 4 * S + std::pow(S, 2) + 2 * std::pow(S, 3)) /
        (8 * std::pow(1 - S, 3));
  e4_ = Th *
        (8 + 12 * S - 152 * std::pow(S, 2) - 308 * std::pow(S, 3) -
         42 * std::pow(S, 4) + 77 * std::pow(S, 5)) /
        (32 * std::pow(1 - S, 5));
  if (StokesOrder_ == 4)
    return;

  // Fifth order coefficients
  a51_ = (-1184 + 32 * S + 13232 * std::pow(S, 2) + 21712 * std::pow(S, 3) +
          20940 * std::pow(S, 4) + 12554 * std::pow(S, 5) -
          500 * std::pow(S, 6) - 3341 * std::pow(S, 7) - 670 * std::pow(S, 8)) /
         (64 * Sh * (3 + 2 * S) * (4 + S) * std::pow(1 - S, 6));
  a53_ =
    (4 * S + 105 * pow(S, 2) + 198 * std::pow(S, 3) - 1376 * std::pow(S, 4) -
     1302 * std::pow(S, 5) - 117 * std::pow(S, 6) + 58 * std::pow(S, 7)) /
    (32 * Sh * (3 + 2 * S) * std::pow(1 - S, 6));
  a55_ = (-6 * std::pow(S, 3) + 272 * std::pow(S, 4) - 1552 * std::pow(S, 5) +
          852 * std::pow(S, 6) + 2029 * std::pow(S, 7) + 430 * std::pow(S, 8)) /
         (64 * Sh * (3 + 2 * S) * (4 + S) * std::pow(1 - S, 6));
  b53_ = 9 *
         (132 + 17 * S - 2216 * std::pow(S, 2) - 5897 * std::pow(S, 3) -
          6292 * std::pow(S, 4) - 2687 * std::pow(S, 5) + 194 * std::pow(S, 6) +
          467 * std::pow(S, 7) + 82 * std::pow(S, 8)) /
         (128 * (3 + 2 * S) * (4 + S) * std::pow(1 - S, 6));
  b55_ = 5 *
         (300 + 1579 * S + 3176 * std::pow(S, 2) + 2949 * std::pow(S, 3) +
          1188 * std::pow(S, 4) + 675 * std::pow(S, 5) + 1326 * std::pow(S, 6) +
          827 * std::pow(S, 7) + 130 * std::pow(S, 8)) /
         (384 * (3 + 2 * S) * (4 + S) * std::pow(1 - S, 6));
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
  c_ = (c0_ + std::pow(eps_, 2) * c2_ + std::pow(eps_, 4) * c4_) *
       std::sqrt(g_ / k_);
  Q_ = c_ * waterdepth_ * std::sqrt(std::pow(k_, 3) / g_) +
       d2_ * std::pow(eps_, 2) +
       d4_ * std::pow(eps_, 4) * std::sqrt(g_ / std::pow(k_, 3));
  cs_ = c_ - Q_;
  period_ = length_ / c_;
  omega_ = c_ * k_;
  return;
}

double
MotionWavesKernel::my_cosh_cos(int i, int j, double phase)
{
  double D=0.0;
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

  return std::pow(eps_, i) * D * j * k_ * std::cosh(j * k_ * waterdepth_) *
         std::cos(j * phase);
}

double
MotionWavesKernel::my_sinh_sin(int i, int j, double phase)
{
  double D=0.0;
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

  return std::pow(eps_, i) * D * j * k_ * std::sinh(j * k_ * waterdepth_) *
         std::sin(j * phase);
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

void
MotionWavesKernel::post_compute_geometry(
  stk::mesh::BulkData& bulk,
  stk::mesh::PartVector& partVec,
  stk::mesh::PartVector& partVecBc,
  bool& computedMeshVelDiv)
{
  if (computedMeshVelDiv)
    return;

  // compute divergence of mesh velocity
  VectorFieldType* meshVelocity =
    bulk.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_velocity");

  ScalarFieldType* meshDivVelocity =
    bulk.mesh_meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "div_mesh_velocity");
 
      stk::mesh::field_fill(0.0, *meshDivVelocity);
  compute_vector_divergence(
    bulk, partVec, partVecBc, meshVelocity, meshDivVelocity, true);
  computedMeshVelDiv = true;
}

} // namespace nalu
} // namespace sierra
