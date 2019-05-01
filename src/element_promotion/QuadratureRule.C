/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/QuadratureRule.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_SerialDenseVector.hpp>

#include <cmath>
#include <vector>
#include <tuple>
#include <iostream>

namespace sierra{
namespace nalu{

std::pair<Teuchos::SerialDenseVector<int, double>, Teuchos::SerialDenseVector<int, double>>
jacobi_recursion_coefficients(
    const double alpha,
    const double beta,
    const int order)
{
  int N = order;
  Teuchos::SerialDenseVector<int, double> a(N);
  Teuchos::SerialDenseVector<int, double> b(N);

  double nu = (beta - alpha) / (alpha + beta + 2.0);
  double mu = std::pow(2.0, alpha + beta + 1.0) * std::tgamma(alpha + 1.0)
              * std::tgamma(beta + 1.0) / std::tgamma(alpha + beta + 2.0);
  double nab;
  double sqdif = beta * beta - alpha * alpha;

  a[0] = nu;
  b[0] = mu;

  if (N > 1) {
    for (int n = 1; n < N; ++n) {
      nab = 2 * n + alpha + beta;
      a[n] = sqdif / (nab * (nab + 2));
    }

    b[1] = 4.0 * (alpha + 1.0) * (beta + 1.0)
                    / (std::pow(alpha + beta + 2.0, 2) * (alpha + beta + 3.0));

    if (N > 2) {
      for (int n = 2; n < N; ++n) {
        nab = 2 * n + alpha + beta;
        b[n] = 4.0 * (n + alpha) * (n + beta) * n * (n + alpha + beta)
                        / (nab * nab * (nab + 1.0) * (nab - 1.0));
      }
    }
  }
  return std::make_pair(a, b);
}
//--------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
gauss_legendre_rule(int order)
{
  /*
   * Returns a pair of abscissae and weights for the usual Gauss-Legendre
   * quadrature rule.
   *
   * Based on the Golub-Welsch algorithm.  Implementation is based on the
   * implementation in Trilinos's rol package, which in turn is based on
   * polylib's implementation
   */

  int INFO;
  const int N = order;
  const char COMPZ = 'I';

  Teuchos::SerialDenseVector<int, double> D(N);
  Teuchos::SerialDenseVector<int, double> b(N);
  Teuchos::SerialDenseVector<int, double> E(N);
  Teuchos::SerialDenseVector<int, double> work(4 * N);
  Teuchos::SerialDenseMatrix<int, double> Z(N, N);

  std::tie(D, b) = jacobi_recursion_coefficients(0.0, 0.0, order);

  for (int i = 0; i < N - 1; ++i) {
    E[i] = std::sqrt(b[i + 1]);
  }

  Teuchos::LAPACK<int, double>().STEQR(
    COMPZ, N, D.values(), E.values(),
    Z.values(), N, work.values(), &INFO
  );

  std::vector<double> x(N);
  std::vector<double> w(N);
  for (int i = 0; i < N; ++i) {
    x[i] = D(i);
    w[i] = b[0] * Z(0, i) * Z(0, i);
  }
  return std::make_pair(x, w);
}
//--------------------------------------------------------------------
std::pair<Teuchos::SerialDenseVector<int, double>, Teuchos::SerialDenseVector<int, double>>
coefficients_for_lobatto(
  int order,
  double xl1,
  double xl2)
{
  const int N = order - 1;

  Teuchos::SerialDenseVector<int, double> en(N);
  Teuchos::SerialDenseVector<int, double> g(N);
  Teuchos::SerialDenseMatrix<int, double> M(N, N);
  Teuchos::SerialDenseSolver<int, double> solver;
  Teuchos::SerialDenseVector<int, double> a;
  Teuchos::SerialDenseVector<int, double> b;
  std::tie(a, b) = jacobi_recursion_coefficients(0.0, 0.0, N);

  // Nth canonical vector
  en[N - 1] = 1;

  for (int i = 0; i < N; ++i) {
    M(i, i) = a[i] - xl1;
  }

  for (int i = 0; i < N - 1; ++i) {
    double offdiag = std::sqrt(b[i + 1]);
    M(i + 1, i) = offdiag;
    M(i, i + 1) = offdiag;
  }

  solver.setMatrix(Teuchos::rcp(&M, false));
  solver.setVectors(Teuchos::rcp(&g, false), Teuchos::rcp(&en, false));
  solver.solve();
  double g1 = g[N - 1];

  M.putScalar(0.0);
  en.putScalar(0.0);
  g.putScalar(0.0);
  en[N - 1] = 1;
  for (int i = 0; i < N; ++i) {
    M(i, i) = a[i] - xl2;
  }

  for (int i = 0; i < N - 1; ++i) {
    double offdiag = std::sqrt(b[i + 1]);
    M(i + 1, i) = offdiag;
    M(i, i + 1) = offdiag;
  }

  solver.setMatrix(Teuchos::rcp(&M, false));
  solver.setVectors(Teuchos::rcp(&g, false), Teuchos::rcp(&en, false));
  solver.solve();
  double g2 = g[N - 1];

  Teuchos::SerialDenseVector<int, double> amod(N + 1);
  Teuchos::SerialDenseVector<int, double> bmod(N + 1);

  for (int i = 0; i < N; ++i) {
    amod[i] = a[i];
    bmod[i] = b[i];
  }
  amod[N] = (g1 * xl2 - g2 * xl1) / (g1 - g2);
  bmod[N] = (xl2 - xl1) / (g1 - g2);

  return std::make_pair(amod, bmod);
}
//--------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
gauss_lobatto_legendre_rule(
  int order,
  double xleft,
  double xright)
{
  /*
   * Returns a pair of abscissae and weights for the usual Gauss-Legendre
   * quadrature rule.
   *
   * Based on the modified Golub-Welsch algorithm.  Implementation is based on the
   * implementation in Trilinos's rol package, which in turn is based on
   * polylib's implementation
   */

  int INFO;
  const int N = order;
  const char COMPZ = 'I';

  if (order == 1) {
    return std::make_pair(std::vector<double>{xleft, xright}, std::vector<double>{1.0, 1.0});
  }

  Teuchos::SerialDenseVector<int, double> D(N);
  Teuchos::SerialDenseVector<int, double> b(N);
  Teuchos::SerialDenseVector<int, double> E(N);
  Teuchos::SerialDenseVector<int, double> work(4 * N);
  Teuchos::SerialDenseMatrix<int, double> Z(N, N);

  std::tie(D, b) = coefficients_for_lobatto(order, xleft, xright);

  for (int i = 0; i < N - 1; ++i) {
    E[i] = std::sqrt(b[i + 1]);
  }

  Teuchos::LAPACK<int, double>().STEQR(
    COMPZ, N, D.values(), E.values(),
    Z.values(), N, work.values(), &INFO
  );

  std::vector<double> x(N);
  std::vector<double> w(N);
  for (int i = 0; i < N; ++i) {
    x[i] = D(i);
    w[i] = b[0] * Z(0, i) * Z(0, i);
  }

  //remove machine eps from end points
  x[0] = xleft;
  x[N - 1] = xright;

  return std::make_pair(x, w);
}
//--------------------------------------------------------------------
Teuchos::SerialDenseVector<int, double>
subinterval_weights_for_fixed_abscissae(
  std::vector<double> fixedAbscissae,
  double xleft,
  double xright)
{
  /*
   * The goal of this function is to produce quadrature weights
   * for integration of a polynomial over a subinterval,
   * (xleft, xright) \subset (-1,1)
   * given fixed abscissae positioned anywhere in (-1,1)
   *
   * The motivation is that, in CVFEM, we have to integrate
   * over many subintervals quickly.  By making the abscissae
   * of the quadrature rule independent of the subinterval itself,
   * we can potentially speed-up assembly while maintaining
   * exact integration.
   *
   * The actual algorithm is a moment-matching algorithm
   * (e.g. Hildebrand, Introduction to Numerical Analysis, 1974)
   * with integration of the monomials over the subinterval specified
   * by the xleft, xright parameters
   */

  const int nrows = fixedAbscissae.size();
  Teuchos::SerialDenseMatrix<int, double> weightLHS(nrows, nrows);
  for (int j = 0; j < nrows; ++j) {
    for (int i = 0; i < nrows; ++i) {
      weightLHS(i, j) = std::pow(fixedAbscissae[j], i);
    }
  }

  // each node has a separate RHS
  Teuchos::SerialDenseVector<int, double> weightRHS(nrows);
  for (int i = 0; i < nrows; ++i) {
    weightRHS(i) = (std::pow(xright, i + 1) - std::pow(xleft, i + 1)) / (i + 1.0);
  }

  Teuchos::SerialDenseSolver<int, double> solver;
  Teuchos::SerialDenseVector<int, double> quadratureWeights(nrows);
  solver.setMatrix(Teuchos::rcp(&weightLHS, false));
  solver.setVectors(
    Teuchos::rcp(&quadratureWeights, false),
    Teuchos::rcp(&weightRHS, false)
  );
  solver.solve();

  return quadratureWeights;
}
//--------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
SGL_quadrature_rule(int order, const double* scsEndLocations)
{
  /*
   * Produces the weights for a fixed
   * N-point (order) integration rule over a fixed N-number
   * of subintervals, the locations of which are provided by scsEndLocations
   *
   * This is the special "Segmented Gauss Lobatto" quadrature rule designed for CVFEM
   *
   * Output is the 1-dimensional fixed abscissae with the corresponding
   * NxN weights
   */
  int N = order;

  std::vector<double> fixedAbscissae = gauss_lobatto_legendre_rule(N).first;

  std::vector<double> weightTensor(N*N);
  Teuchos::SerialDenseVector<int, double> scvWeights(N);
  for (int j = 0; j < N; ++j) {
    scvWeights = subinterval_weights_for_fixed_abscissae(
      fixedAbscissae,
      scsEndLocations[j], scsEndLocations[j+1]
    );

    // save to a standard container
    for (int i = 0; i < N; ++i) {
      weightTensor[i + j * N] = scvWeights[i];
    }
  }

  return std::make_pair(fixedAbscissae, weightTensor);
}
//--------------------------------------------------------------------
std::vector<double> pad_end_points(const std::vector<double>& x, double xleft, double xright)
{
  std::vector<double> padded_x(x.size()+2);
  padded_x[0] = xleft;
  for (unsigned j = 0; j < x.size();++j) {
    padded_x[j+1] = x[j];
  }
  padded_x[x.size()+1] = xright;

  return padded_x;
}

std::vector<double> pad_end_points(int n, const double* x, double xleft, double xright)
{
  std::vector<double> padded_x(n+2);
  padded_x[0] = xleft;
  for (int j = 0; j < n;++j) {
    padded_x[j+1] = x[j];
  }
  padded_x[n+1] = xright;

  return padded_x;
}


}  // namespace nalu
}  // namespace sierra
