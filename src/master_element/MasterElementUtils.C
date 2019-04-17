/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <master_element/MasterElementUtils.h>

#include <element_promotion/LagrangeBasis.h>
#include <master_element/TensorOps.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sierra{
namespace nalu{

  bool isoparameteric_coordinates_for_point_3d(
    sierra::nalu::LagrangeBasis& basis,
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT pointCoord,
    double* POINTER_RESTRICT isoParCoord,
    std::array<double,3> initialGuess,
    int maxIter,
    double tol,
    double deltaLimit)
  {
    int nNodes = basis.num_nodes();
    constexpr int dim = 3;
    std::array<double, dim> guess = initialGuess;
    std::array<double, dim> delta;
    int iter = 0;

    do {
      // interpolate coordinate at guess
      const auto& weights = basis.point_interpolation_weights(guess.data());

      std::array<double, dim> error_vec;
      error_vec[0] = pointCoord[0] - ddot(weights.data(), elemNodalCoords + 0 * nNodes, nNodes);
      error_vec[1] = pointCoord[1] - ddot(weights.data(), elemNodalCoords + 1 * nNodes, nNodes);
      error_vec[2] = pointCoord[2] - ddot(weights.data(), elemNodalCoords + 2 * nNodes, nNodes);

      // update guess along gradient of mapping from physical-to-reference coordinates

      // transpose of the jacobian of the forward mapping
      const auto& deriv = basis.point_derivative_weights(guess.data());
      std::array<double, dim * dim> jact{};
      for(int j = 0; j < nNodes; ++j) {
        jact[0] += deriv(j, 0) * elemNodalCoords[j + 0 * nNodes];
        jact[1] += deriv(j, 1) * elemNodalCoords[j + 0 * nNodes];
        jact[2] += deriv(j, 2) * elemNodalCoords[j + 0 * nNodes];

        jact[3] += deriv(j, 0) * elemNodalCoords[j + 1 * nNodes];
        jact[4] += deriv(j, 1) * elemNodalCoords[j + 1 * nNodes];
        jact[5] += deriv(j, 2) * elemNodalCoords[j + 1 * nNodes];

        jact[6] += deriv(j, 0) * elemNodalCoords[j + 2 * nNodes];
        jact[7] += deriv(j, 1) * elemNodalCoords[j + 2 * nNodes];
        jact[8] += deriv(j, 2) * elemNodalCoords[j + 2 * nNodes];
      }

      // apply its inverse on the error vector
      solve33(jact.data(), error_vec.data(), delta.data());

      // update guess.  Break if update is running away & report failure
      if (vecnorm_sq3(delta.data()) > deltaLimit) {
        iter = maxIter;
        break;
      }

      guess[0] += delta[0];
      guess[1] += delta[1];
      guess[2] += delta[2];

    } while (vecnorm_sq3(delta.data()) > tol && (++iter < maxIter));

    isoParCoord[0] = guess[0];
    isoParCoord[1] = guess[1];
    isoParCoord[2] = guess[2];

    return (iter < maxIter);
  }

  bool isoparameteric_coordinates_for_point_2d(
    sierra::nalu::LagrangeBasis& basis,
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT pointCoord,
    double* POINTER_RESTRICT isoParCoord,
    std::array<double,2> initialGuess,
    int maxIter,
    double tol,
    double deltaLimit)
  {
    int nNodes = basis.num_nodes();
    constexpr int dim = 2;
    std::array<double, dim> guess = initialGuess;
    std::array<double, dim> delta;
    int iter = 0;

    do {
      // interpolate coordinate at guess
      const auto& weights = basis.point_interpolation_weights(guess.data());

      std::array<double, dim> error_vec;
      error_vec[0] = pointCoord[0] - ddot(weights.data(), elemNodalCoords + 0 * nNodes, nNodes);
      error_vec[1] = pointCoord[1] - ddot(weights.data(), elemNodalCoords + 1 * nNodes, nNodes);

      // update guess along gradient of mapping from physical-to-reference coordinates

      // transpose of the jacobian of the forward mapping
      const auto& deriv = basis.point_derivative_weights(guess.data());
      std::array<double, dim * dim> jact{};
      for(int j = 0; j < nNodes; ++j) {
        jact[0] += deriv(j,0) * elemNodalCoords[j + 0 * nNodes];
        jact[1] += deriv(j,1) * elemNodalCoords[j + 0 * nNodes];
        jact[2] += deriv(j,0) * elemNodalCoords[j + 1 * nNodes];
        jact[3] += deriv(j,1) * elemNodalCoords[j + 1 * nNodes];
      }

      // apply its inverse on the error vector
      solve22(jact.data(), error_vec.data(), delta.data());

      // update guess.  Break if update is running away & report failure
      if (vecnorm_sq2(delta.data()) > deltaLimit) {
        iter = maxIter;
        break;
      }

      guess[0] += delta[0];
      guess[1] += delta[1];
    } while (vecnorm_sq2(delta.data()) > tol && (++iter < maxIter));

    isoParCoord[0] = guess[0];
    isoParCoord[1] = guess[1];

    return (iter < maxIter);
  }

}
}
