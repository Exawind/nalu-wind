#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>

namespace sierra {
namespace nalu {

//============================================================================
/**
 *  This file contains generic mathematical functions that are needed at
 *  run-time by our Converter classes.  An implementation of erf() and erfc()
 *  are available in math.h, but we would prefer to only link with C++
 *  headers.  (These functions are not available in cmath.h.)
 */

//============================================================================
/** Compute the error function using series solutions.  This returns
 *  values with relative error everywhere less than 1.58e-12 when compared
 *  to the glibc implementation of erf().  This error occurs at x=3.5
 *  where the solution switches to an asymptotic expansion and recovers
 *  ~1.e-16 error above about 4.2 and for all values below 3.5.
 */
double errorf(const double x);

//============================================================================
/** Compute the complementary error function using series solutions.
 */
double errorfc(const double x);

//============================================================================
/** Compute the inverse error function using Newton's method.  Accuracy
 *  will be full machine precision for all inputs except values just a
 *  couple significant figures below 1.0, where relative accuracy will
 *  fall to about 1.e-5.
 */
double inv_errorf(const double x);

//============================================================================
/** Compute the quantity:
 *
 *    F_chi(Z) = exp( -2 * ( inv_errorf(1 - 2*Z)^2 ) )
 *
 *  which is a sub-component of an exact solution of scalar dissipation
 *  rate from laminar flamelet theory:
 *
 *    Chi = Chi_max * F_chi(Z)
 *
 *  Note that this function only returns F_chi(Z) and not Chi itself.
 */
double F_chi(const double Z);

//============================================================================
/** Class to wrap the F_chi() function with an interface that is
 *  compatible with our Functor implementation.
 */
class FChi
{
public:
  FChi() {};
  ~FChi() {};

  double query(const double Z);
};

//============================================================================
/** Compute the quantity F_gamma (and not Gamma itself)
 */
double F_gamma(const double Z, const double Z_st);

double F_gamma(
  const std::vector<double>& Z,
  std::vector<std::vector<double>> Z_st, // local copy
  std::vector<double> gamma_mas_st);     // local copy

//============================================================================
/** Class to wrap the F_gamma() function with an interface that is
 *  compatible with our Functor implementation.
 */
class FGamma
{
public:
  explicit FGamma(const int nMixFrac) { zBuf_.resize(nMixFrac, 0.0); }
  ~FGamma() {};

  void setZStoich(const std::vector<std::vector<double>>& zStoich)
  {
    zStoich_ = zStoich;
  }
  void setGammaMaxStoich(const std::vector<double>& gammaMaxStoich)
  {
    gammaMaxStoich_ = gammaMaxStoich;
  }
  double query(const double* Z) const;

private:
  mutable std::vector<double> zBuf_;
  std::vector<std::vector<double>> zStoich_;
  std::vector<double> gammaMaxStoich_;
};

} // namespace nalu
} // namespace sierra

#endif
