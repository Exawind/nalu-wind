#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include "EigenDecomposition.h"

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include "UnitTestUtils.h"


#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/MasterElement.h>

#include <fstream>



namespace {

    std::random_device rd; 
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> rand(0.0, 1.0);
 
    double a11 = rand(rng);
    double a22 = rand(rng);
    double a33 = rand(rng);
    double a12 = rand(rng);
    double a13 = rand(rng);
    double a23 = rand(rng);

    double b11 = rand(rng);
    double b22 = rand(rng);
    double b33 = rand(rng);

static double A3d_rand[3][3] = {
  {a11, a12, a13}, {a12, a22, a23}, {a13, a23, a33},
};

static double A3d_fixed[3][3] = {
  {0.0562500000000000,  0.1875000000000000,  0.0125000000000000},
  {0.1875000000000000,  0.0093750000000000, -0.4218750000000000},
  {0.0125000000000000, -0.4218750000000000,  0.0031250000000000},
};

static double A2d_rand[2][2] = {
  {a11, a12}, {a12, a22},
};

static double A2d_fixed[2][2] = {
  { 0.0562500000000000, -0.0187500000000000},  
  {-0.018750000000000,   0.0093750000000000}, 
};

DoubleType A3d_rand_simd[3][3];
DoubleType A3d_fixed_simd[3][3];
DoubleType A2d_rand_simd[2][2];
DoubleType A2d_fixed_simd[2][2];

} // namespace

// This tests whether the correct eigenvalues are obtained 
TEST(TestEigen, testeigendecomp3d)
{
  double Q_[3][3], D_[3][3];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A3d_fixed, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[3] = {
      0.056736539229635605, 0.46782517604126655, -0.45581171527090225};

  for (unsigned j = 0 ; j < 3; ++j) {
    EXPECT_NEAR(D_[j][j], lambda_gold[j], tol);
  }
}

TEST(TestEigen, testeigendecomp2d)
{
  double Q_[2][2], D_[2][2];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A2d_fixed, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[2] = {
      0.06282714486296648, 0.0027978551370335227};

  for (unsigned j = 0; j < 2; ++j) {
    EXPECT_NEAR(D_[j][j], lambda_gold[j], tol);
  }
}

// This tests that the eignevalue decomposition and reconstruction
// returns the original matrix
TEST(TestEigen, testeigendecompandreconstruct3d)
{
  double b_[3][3], Q_[3][3], D_[3][3];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A3d_rand, Q_, D_);
  sierra::nalu::EigenDecomposition::reconstruct_matrix_from_decomposition(D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0 ; j < 3; ++j) {
    for (unsigned i = 0; i < 3; ++i) {
      EXPECT_NEAR(b_[i][j], A3d_rand[i][j], tol);
    }
  }
}

// This tests that the eignevalue decomposition and reconstruction
// returns the original matrix
TEST(TestEigen, testeigendecompandreconstruct2d)
{
  double b_[2][2], Q_[2][2], D_[2][2];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A2d_rand, Q_, D_);
  sierra::nalu::EigenDecomposition::reconstruct_matrix_from_decomposition(D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;
  
  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0 ; j < 2; ++j) {
    for (unsigned i = 0; i < 2; ++i) {
      EXPECT_NEAR(b_[i][j], A2d_rand[i][j], tol);
    }
  }
}


// SIMD tests

// This tests whether the correct eigenvalues are obtained 
TEST(TestEigen, testeigendecomp3d_simd)
{
  DoubleType Q_[3][3], D_[3][3];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    A3d_fixed_simd[0][0]._data[j] = A3d_fixed[0][0]*(j + 1);
    A3d_fixed_simd[0][1]._data[j] = A3d_fixed[0][1]*(j + 1);
    A3d_fixed_simd[0][2]._data[j] = A3d_fixed[0][2]*(j + 1);
    A3d_fixed_simd[1][1]._data[j] = A3d_fixed[1][1]*(j + 1);
    A3d_fixed_simd[1][2]._data[j] = A3d_fixed[1][2]*(j + 1);
    A3d_fixed_simd[2][2]._data[j] = A3d_fixed[2][2]*(j + 1);
  }
  A3d_fixed_simd[1][0] = A3d_fixed_simd[0][1];
  A3d_fixed_simd[2][0] = A3d_fixed_simd[0][2];
  A3d_fixed_simd[2][1] = A3d_fixed_simd[1][2];
 

  sierra::nalu::EigenDecomposition::sym_diagonalize(A3d_fixed_simd, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[3] = {
      0.056736539229635605, 0.46782517604126655, -0.45581171527090225};

  for (unsigned j = 0; j < 3; ++j) {
    for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
      EXPECT_NEAR(stk::simd::get_data(D_[j][j], is), (is+1)*lambda_gold[j], tol);
    }
  }
}

TEST(TestEigen, testeigendecomp2d_simd)
{
  DoubleType Q_[2][2], D_[2][2];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    A2d_fixed_simd[0][0]._data[j] = A2d_fixed[0][0]*(j + 1);
    A2d_fixed_simd[0][1]._data[j] = A2d_fixed[0][1]*(j + 1);
    A2d_fixed_simd[1][1]._data[j] = A2d_fixed[1][1]*(j + 1);
  }
  A2d_fixed_simd[1][0] = A2d_fixed_simd[0][1];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A2d_fixed_simd, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[2] = {
      0.06282714486296648, 0.0027978551370335227};

  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
      EXPECT_NEAR(stk::simd::get_data(D_[j][j], is), (is+1)*lambda_gold[j], tol);
    }
  }
}

// This tests that the eigenvalue decomposition and reconstruction
// returns the original matrix
TEST(TestEigen, testeigendecompandreconstruct3d_simd)
{
  DoubleType b_[3][3], Q_[3][3], D_[3][3];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    if (j % 2 == 0) {
      A3d_rand_simd[0][0]._data[j] = a11*(j + 1);
      A3d_rand_simd[0][1]._data[j] = a12*(j + 1);
      A3d_rand_simd[0][2]._data[j] = a13*(j + 1);
      A3d_rand_simd[1][1]._data[j] = a22*(j + 1);
      A3d_rand_simd[1][2]._data[j] = a23*(j + 1);
      A3d_rand_simd[2][2]._data[j] = a33*(j + 1);
    }
    // Test if some of the simd entries are already diagonal
    else {
      A3d_rand_simd[0][0]._data[j] = b11*(j + 1);
      A3d_rand_simd[0][1]._data[j] = 0.0;
      A3d_rand_simd[0][2]._data[j] = 0.0;
      A3d_rand_simd[1][1]._data[j] = b22*(j + 1);
      A3d_rand_simd[1][2]._data[j] = 0.0;
      A3d_rand_simd[2][2]._data[j] = b33*(j + 1);
    }
  }
  A3d_rand_simd[1][0] = A3d_rand_simd[0][1];
  A3d_rand_simd[2][0] = A3d_rand_simd[0][2];
  A3d_rand_simd[2][1] = A3d_rand_simd[1][2];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A3d_rand_simd, Q_, D_);
  sierra::nalu::EigenDecomposition::reconstruct_matrix_from_decomposition(D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0; j < 3; ++j) {
    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
        EXPECT_NEAR(stk::simd::get_data(b_[i][j], is), stk::simd::get_data(A3d_rand_simd[i][j], is), tol);
      }
    }
  }
}

TEST(TestEigen, testeigendecompandreconstruct2d_simd)
{
  DoubleType b_[2][2], Q_[2][2], D_[2][2];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    if (j % 2 == 0) {
      A2d_rand_simd[0][0]._data[j] = a11*(j + 1);
      A2d_rand_simd[0][1]._data[j] = a12*(j + 1);
      A2d_rand_simd[1][1]._data[j] = a22*(j + 1);
    }
    // Test if some of the simd entries are already diagonal
    else {
      A2d_rand_simd[0][0]._data[j] = b11*(j + 1);
      A2d_rand_simd[0][1]._data[j] = 0.0;
      A2d_rand_simd[1][1]._data[j] = b22*(j + 1);
    }
  }
  A2d_rand_simd[1][0] = A2d_rand_simd[0][1];

  sierra::nalu::EigenDecomposition::sym_diagonalize(A2d_rand_simd, Q_, D_);
  sierra::nalu::EigenDecomposition::reconstruct_matrix_from_decomposition(D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0 ; j < 2; ++j) {
    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
        EXPECT_NEAR(stk::simd::get_data(b_[i][j], is), stk::simd::get_data(A2d_rand_simd[i][j], is), tol);
      }
    }
  }
}

