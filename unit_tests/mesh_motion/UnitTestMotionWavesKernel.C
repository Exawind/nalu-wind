#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MotionWavesKernel.h"
#include "mesh_motion/NgpMotion.h"

#include "UnitTestRealm.h"

namespace {

  const double testTol = 1e-14;
  const double CoeffTol = 1e-4; //Reduced tolerance for the Stokes Coeffs 

  std::vector<double> transform(
    const sierra::nalu::mm::TransMatType& transMat,
    const sierra::nalu::mm::ThreeDVecType& xyz )
  {
    std::vector<double> transCoord(3,0.0);

    // perform matrix multiplication between transformation matrix
    // and original coordinates to obtain transformed coordinates
    for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++) {
      transCoord[d] = transMat[d*sierra::nalu::mm::matSize+0]*xyz[0]
                     +transMat[d*sierra::nalu::mm::matSize+1]*xyz[1]
                     +transMat[d*sierra::nalu::mm::matSize+2]*xyz[2]
                     +transMat[d*sierra::nalu::mm::matSize+3];
    }

    return transCoord;
  }

}

TEST(meshMotion, airy_wave)
{
  // create a yaml node describing rotation
  const std::string Airy_Wave_info =
    "wave_model: Airy            \n"
    "wave_height: 0.1            \n"
    "wave_length: 3.14159265359  \n"
    "water_depth: 0.376991       \n"
    "mesh_damping_length: 1.     \n"
    ;
  YAML::Node Airy_Wave_node = YAML::Load(Airy_Wave_info);

  // initialize the mesh Wave motion class
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  sierra::nalu::MotionWavesKernel MotionWavesKernel(realm.meta_data(),Airy_Wave_node);

  // build transformation
  const double time = 1.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5,1.5,0.};

  sierra::nalu::mm::TransMatType transMat = MotionWavesKernel.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_z = 0.0053635368158730;

  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

}


TEST(meshMotion, stokes_coefficients)
{
  const std::string Stokes_Wave_info =
    "wave_model: Stokes          \n"
    "Stokes_order: 5             \n"
    "wave_height: 0.25           \n"
    "wave_length: 3.14159265359  \n"
    "water_depth: 0.376991       \n"
    "mesh_damping_length: 1.      \n"
    ;
  
  YAML::Node Stokes_Wave_node = YAML::Load(Stokes_Wave_info);
  // initialize the mesh Wave motion class
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  
  sierra::nalu::MotionWavesKernel MotionWavesKernel(realm.meta_data(),Stokes_Wave_node);
  // Coefficients values as presented in table 2 of Fenton 1985
  sierra::nalu::MotionWavesKernel::StokesCoeff stokes_coeff;
  MotionWavesKernel.get_StokesCoeff(&stokes_coeff);

  const double gold_A11 = 1.208490;  
  const double gold_A22 = 0.799840;
  const double gold_A31 = -9.105340;
  const double gold_A33 = 0.368275;
  const double gold_A42 = -12.196150;
  const double gold_A44 = 0.058723;
  const double gold_A51 = 108.46831725;
  const double gold_A53 = -6.941756;
  const double gold_A55 = -0.074979;
  const double gold_B22 = 2.502414;
  const double gold_B31 = -5.731666;
  const double gold_B42 = -32.407508;
  const double gold_B44 = 14.033758;
  const double gold_B53 = -103.44536875;
  const double gold_B55 = 37.200027;
  const double gold_C0  = 0.798448;
  const double gold_C2  = 1.940215;
  const double gold_C4  = -12.970403;
  const double gold_D2  = -0.626215;
  const double gold_D4  = 3.257104;
  const double gold_E2  = 1.781926;
  const double gold_E4  = -11.573657;

  EXPECT_NEAR(gold_A11,stokes_coeff.a11, CoeffTol);
  EXPECT_NEAR(gold_A22,stokes_coeff.a22, CoeffTol);
  EXPECT_NEAR(gold_A31,stokes_coeff.a31, CoeffTol);
  EXPECT_NEAR(gold_A33,stokes_coeff.a33, CoeffTol);
  EXPECT_NEAR(gold_A42,stokes_coeff.a42, CoeffTol);
  EXPECT_NEAR(gold_A44,stokes_coeff.a44, CoeffTol);
  EXPECT_NEAR(gold_A51,stokes_coeff.a51, CoeffTol);
  EXPECT_NEAR(gold_A53,stokes_coeff.a53, CoeffTol);
  EXPECT_NEAR(gold_A55,stokes_coeff.a55, CoeffTol);
  EXPECT_NEAR(gold_B22,stokes_coeff.b22, CoeffTol);
  EXPECT_NEAR(gold_B31,stokes_coeff.b31, CoeffTol);
  EXPECT_NEAR(gold_B42,stokes_coeff.b42, CoeffTol);
  EXPECT_NEAR(gold_B44,stokes_coeff.b44, CoeffTol);
  EXPECT_NEAR(gold_B53,stokes_coeff.b53, CoeffTol);
  EXPECT_NEAR(gold_B55,stokes_coeff.b55, CoeffTol);
  EXPECT_NEAR(gold_C0, stokes_coeff.c0, CoeffTol);
  EXPECT_NEAR(gold_C2, stokes_coeff.c2, CoeffTol);
  EXPECT_NEAR(gold_C4, stokes_coeff.c4, CoeffTol);
  EXPECT_NEAR(gold_D2, stokes_coeff.d2, CoeffTol);
  EXPECT_NEAR(gold_D4, stokes_coeff.d4, CoeffTol);
  EXPECT_NEAR(gold_E2, stokes_coeff.e2, CoeffTol);
  EXPECT_NEAR(gold_E4, stokes_coeff.e4, CoeffTol);

}
