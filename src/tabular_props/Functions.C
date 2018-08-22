#include <tabular_props/Functions.h>
#include <property_evaluator/HDF5TablePropAlgorithm.h>

#include <stk_util/util/ReportHandler.hpp>

#include <stdexcept>
#include <cmath>
#include <algorithm>

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
double errorf( const double x )
{
  const double EPS = 1.e-15;
  const double SQRTPI = 1.7724538509055160273;

  const double x2 = x * x;
  if ( std::abs(x) < 3.5 ) {
    // Power series for small arguments
    double er = 1.0;
    double r = 1.0;
    for ( int i = 1; i <= 50; ++i ) {
      r = r * x2 / ( i + 0.5 );
      er = er + r;
      if ( r <= ( er * EPS ) ) {
        break;
      }
    }
    double c0 = 2.0 * x * std::exp(-x2) / SQRTPI;
    return c0 * er;
  }
  else {
    // Asymptotic expansion for large arguments
    double er = 1.0;
    double r = 1.0;
    for ( int i = 1; i <= 12; ++i ) {
      r = -r * ( i - 0.5 ) / x2;
      er = er + r;
    }
    double c0 = std::exp(-x2) / ( std::abs(x) * SQRTPI );
    if ( x >= 0.0 ) {
      return 1.0 - c0 * er;
    }
    else {
      return c0 * er - 1.0;
    }
  }
}

//============================================================================
/** Compute the complementary error function using series solutions.
 */
double errorfc( const double x )
{
  return 1.0 - errorf( x );
}

//============================================================================
/** Compute the inverse error function using Newton's method.  Accuracy
 *  will be full machine precision for all inputs except values just a
 *  couple significant figures below 1.0, where relative accuracy will
 *  fall to about 1.e-5.
 */
double inv_errorf( const double x )
{
  // Set the tolerance and iteration maximum.  This tolerance may look
  // sloppy, but it is enough to lock down full machine precision over
  // most of the input range.  It only falls back to this tolerance when
  // the input is a couple significant figures short of exactly 1.0, and
  // these values are horribly expensive and difficult to achieve and
  // (arguably) unimportant.
  const double TOL = 1.e-5;
  const int IMAX = 20;

  // Get the absolute value of our input function, clipped down to 1.0, and
  // store its sign for adjusting the return value.
  //
  double sign = 1.0;
  double abs_x = x;
  if ( x < 0.0 ) {
    sign = -1.0;
    abs_x = -x;
  }
  abs_x = std::min( abs_x, 1.0 );

  // Set sensible initial guesses to speed convergence, since iteration
  // converges very poorly as we approach 1.0.  Some key points along
  // the way are:
  //
  //  erf( 1.0 ) = 0.842700792949715
  //  erf( 2.0 ) = 0.995322265018953
  //  erf( 3.0 ) = 0.999977909503001
  //  erf( 4.0 ) = 0.999999984582740
  //  erf( 5.0 ) = 0.999999999998463
  //  erf( 5.5 ) = 0.999999999999993
  //
  // Truncate these numbers on the boundaries (do not round) so that
  // we will always approach the solution from the low side.  Otherwise,
  // odds are good that we will jump to a negative solution and iteration
  // will not converge.
  //
  // With these initial guesses, the solution will usually converge in
  // around 3-4 iterations for most input values.  The iteration count
  // will peak at about 13 for input values very near (and including) 1.0.
  //
  //
  double inv_errorf;
  if ( abs_x < 0.84 ) {
    inv_errorf = 0.0;
  }
  else if ( abs_x < 0.995 ) {
    inv_errorf = 1.0;
  }
  else if ( abs_x < 0.99997 ) {
    inv_errorf = 2.0;
  }
  else if ( abs_x < 0.99999998 ) {
    inv_errorf = 3.0;
  }
  else if ( abs_x < 0.999999999998 ) {
    inv_errorf = 4.0;
  }
  else if ( abs_x < 0.99999999999999 ) {
    inv_errorf = 5.0;
  }
  else {
    inv_errorf = 5.5;
  }

  const double C_2_SQRTPI = 1.12837916709551257390;  // 2/sqrt(pi)
  int i;
  for ( i = 0; i < IMAX; ++i ) {
    const double dx = ( C_2_SQRTPI * std::exp( - inv_errorf * inv_errorf ) );
    const double dinv_errorf = (errorf( inv_errorf ) - abs_x) / dx;
    inv_errorf -= dinv_errorf;
    if ( std::abs(dinv_errorf) < TOL ) {
      break;
    }
  }
  return sign * inv_errorf;
}

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
//============================================================================
double F_chi( const double Z )
{
  return std::exp( -2.0 * std::pow( inv_errorf( 1.0 - 2.0 * Z ), 2 ) );
}

//============================================================================
/** Class to wrap the F_chi() function with an interface that is
 *  compatible with our Functor implementation.
 */
//============================================================================
double
FChi::query( const double Z )
{
  return F_chi( Z );
}

//============================================================================
/** Class to wrap the F_gamma() function with an interface that is
 *  compatible with our Functor implementation.
 */
//============================================================================
double
FGamma::query( const double * Z ) const
{
  // Map the raw array values into a std::vector for use in F_gamma().
  for ( unsigned int i = 0; i < zBuf_.size(); ++i ) {
    zBuf_[i] = Z[i];
  }
  return F_gamma( zBuf_, zStoich_, gammaMaxStoich_ );
}

//============================================================================
/** Helper class to simplify 2D coordinate storage and calculations */
//============================================================================
class Coord2D {
 public:
  Coord2D( double xInit = 0.0, double yInit = 0.0 )
    : x( xInit ),
      y( yInit ) {};
  double x;
  double y;
  double magnitude() const { return std::sqrt( x*x +y*y ); }
  double distance( const Coord2D & point ) const {
    return std::sqrt( (x-point.x)*(x-point.x) + (y-point.y)*(y-point.y) );
  }
};

//============================================================================
/** Compute the value of F_gamma (not Gamma itself) for one or two mixture
 *  fractions.  (Three or more mixture fractions are not yet supported.)
 */
//============================================================================
double F_gamma( const std::vector<double> & Zpoint,
                std::vector<std::vector<double> > Z_st,  // local copy
                std::vector<double> gamma_st )           // local copy
{
  // Only works for up to a three-stream problem (for now)
  ThrowRequire( Z_st.size() > 0 && Z_st.size() <= Zpoint.size() );
  ThrowRequire( gamma_st.size() == Z_st.size() );
  const double SMALL = 1.0e-16;
  double value = 0.0;

  //
  // Bail out right away if we're in the unrealizable region
  //
  double Zsum = 0.0;
  for ( unsigned int i = 0; i < Zpoint.size(); ++i ) {
    Zsum += Zpoint[i]; 
  }
  if ( Zsum > 1.0 ) {
    return 0.0;
  }

  if ( Zpoint.size() == 1 ) {
    //
    // One mixture fraction (two stream problem)
    //
    if ( Zpoint[0] <= Z_st[0][0] ) {
      value = gamma_st[0] * Zpoint[0] / Z_st[0][0];
    }
    else {
      value = gamma_st[0] * ( 1.0 - Zpoint[0] ) / ( 1.0 - Z_st[0][0] );
    }

  }
  else if ( Zpoint.size() == 2 ) {
    //
    // Two mixture fractions (three stream problem)
    //
    // The geometric configurations solved below are as follows.  Note that
    // any of the three corners of the realizable region may act as the
    // "origin" of the radial lines used to interpolate values.  Note also
    // that the stoichiometric curve shown is defined by two user-provided
    // stoichiometric points.  If only one point is provided, the
    // stoichiometric line is extended from that point to the opposite corner,
    // where the gamma value is driven to zero.
    //
    //  Z1                            (*) = Bounding line
    //  |                             (\) = Stoichiometric line
    //  |                              Z  = Queried mixture fraction point 
    //  | *                            O  = Origin of radial line through Z
    //  |   *                          S  = Intersection of radial line and
    //  |     *B                            stoichiometric line
    //  |     / *                      B  = Intersection of radial line and
    //  |\   Z    *                         bounding line
    //  | \ /       *
    //  |  S          *
    //  | / \           *
    //  |/   \            *
    //  O-------------------*--- Z0
    // 

    if ( Z_st.size() == 1 ) {
      //
      // One stoichiometric point, so add a second point at the opposite
      // corner with zero gamma
      //
      std::vector<double> newPoint(2, 0.0);
      if ( Z_st[0][0] > SMALL && Z_st[0][1] < SMALL ) {
        newPoint[1] = 1.0;
      }
      else if ( Z_st[0][0] < SMALL && Z_st[0][1] > SMALL ) {
        newPoint[0] = 1.0;
      }
      Z_st.push_back( newPoint );
      gamma_st.push_back( 0.0 );
    }

    
    //
    // Pull the source data into a more convenient structure, to make the
    // code to follow much easier to understand
    //
    const Coord2D Zst1( Z_st[0][0], Z_st[0][1] );
    const Coord2D Zst2( Z_st[1][0], Z_st[1][1] );
    const Coord2D Z( Zpoint[0], Zpoint[1] );

    //
    // Identify the virtual origin (enclosed on two sides by stoichiometric
    // points) and the opposite bounding curve endpoints
    //
    Coord2D O, B1, B2;
    if ( Zst1.x > SMALL && Zst1.y < SMALL ) {
      if ( Zst2.x < SMALL && Zst2.y > SMALL ) {
        O.x  = 0.0; O.y  = 0.0;
        B1.x = 1.0; B1.y = 0.0;
        B2.x = 0.0; B2.y = 1.0;
      }
      else if ( Zst2.x > SMALL && Zst2.y > SMALL ) {
        O.x  = 1.0; O.y  = 0.0;
        B1.x = 0.0; B1.y = 0.0;
        B2.x = 0.0; B2.y = 1.0;
      }
    }
    else {
      O.x  = 0.0; O.y  = 1.0;
      B1.x = 0.0; B1.y = 0.0;
      B2.x = 1.0; B2.y = 0.0;
    } 

    //
    // Coordinates of mixture fraction point (Z), intersection of radial
    // line through origin and mixture fraction point with ridge (R) curve,
    // and intersection of radial line with the bounding (B) curve.
    //
    const bool dxZOZero = (std::abs(Z.x-O.x) < SMALL);
    const bool dyZOZero = (std::abs(Z.y-O.y) < SMALL);
    const bool dxZstZero = (std::abs(Zst1.x-Zst2.x) < SMALL);
    const bool dyZstZero = (std::abs(Zst1.y-Zst2.y) < SMALL);
    const bool dxBZero = (std::abs(B1.x-B2.x) < SMALL);
    const bool dyBZero = (std::abs(B1.y-B2.y) < SMALL);

    const double ZOyxRatio = dxZOZero ? 0.0 : (O.y-Z.y)/(O.x-Z.x);
    const double ZOxyRatio = dyZOZero ? 0.0 : (O.x-Z.x)/(O.y-Z.y);
    const double ZstyxRatio = dxZstZero ? 0.0 : (Zst1.y-Zst2.y)/(Zst1.x-Zst2.x);
    const double ZstxyRatio = dyZstZero ? 0.0 : (Zst1.x-Zst2.x)/(Zst1.y-Zst2.y);
    const double ByxRatio = dxBZero ? 0.0 : (B1.y-B2.y)/(B1.x-B2.x);
    const double BxyRatio = dyBZero ? 0.0 : (B1.x-B2.x)/(B1.y-B2.y);

    //
    // Build up the coordinates of the intersection of the radial line
    // through the origin and the provided mixture fraction point with the
    // stoichiometric line.
    //
    double Sx = 0.0;
    if ( dxZOZero ) {
      Sx = O.x;  // Same x as origin
    }
    else if ( dxZstZero ) {
      Sx = Zst1.x;  // Same x as stoichiometric ridge
    }
    else {
      Sx = (Zst2.y-Z.y+Z.x*ZOyxRatio-Zst2.x*ZstyxRatio)/(ZOyxRatio-ZstyxRatio);
    }

    double Sy = 0.0;
    if ( dyZOZero ) {
      Sy = O.y;  // Same y as origin
    }
    else if ( dyZstZero ) {
      Sy = Zst1.y;  // Same y as stoichiometric ridge
    }
    else {
      Sy = (Zst2.x-Z.x+Z.y*ZOxyRatio-Zst2.y*ZstxyRatio)/(ZOxyRatio-ZstxyRatio);
    }

    const Coord2D S( Sx, Sy );

    //
    // Build up the coordinates of the intersection of the radial line
    // through the origin and the provided mixture fraction point with the
    // bounding line.
    //
    double Bx = 0.0;
    if ( dxZOZero ) {
      Bx = O.x;  // Same x as origin
    }
    else if ( dxBZero ) {
      Bx = B1.x;  // Same x as bounding curve
    }
    else {
      Bx = (B2.y-Z.y+Z.x*ZOyxRatio-B2.x*ByxRatio)/(ZOyxRatio-ByxRatio);
    }

    double By = 0.0;
    if ( dyZOZero ) {
      By = O.y;  // Same y as origin
    }
    else if ( dyBZero ) {
      By = B1.y;  // Same y as bounding curve
    }
    else {
      By = (B2.x-Z.x+Z.y*ZOxyRatio-B2.y*BxyRatio)/(ZOxyRatio-BxyRatio);
    }

    const Coord2D B( Bx, By );


    //
    // Distances between points and the origin
    //
    const double distZO = Z.distance( O );
    const double distSO = S.distance( O );
    const double distBO = B.distance( O );

    //
    // Gamma value at intersection of radial line with stoichiometric line
    // (linear blend between end-point values)
    //
    const double gammaS = gamma_st[0] + (gamma_st[1]-gamma_st[0]) *
                          Zst1.distance( S ) / Zst1.distance( Zst2 );

    //
    // Gamma value at given mixture fraction point
    //
    if ( distZO <= distSO ) {
      // Inside stoichiometric line
      value = (distSO > SMALL) ? gammaS * distZO / distSO : 0.0;
    }
    else if ( distZO <= distBO ) {
      // Between stoichiometric line and bounding line
      const double distZB = Z.distance( B );
      const double distSB = S.distance( B );
      value = (distSB > SMALL) ? gammaS * distZB / distSB : gammaS;
    }
    else {
      // Outside bounding line (unrealizable region)
      value = 0.0;
    }

  }
  else {
    throw std::runtime_error("No more than two mixture fractions are supported in F_gamma()");
  }

  //
  // Renormalize result so that the maximum value is unity
  //
  double norm = std::abs( gamma_st[0] ); 
  unsigned int idxGammaMax = 0;
  for ( unsigned int i = 1; i < gamma_st.size(); ++i ) {
    if ( std::abs( gamma_st[i] ) > norm ) {
      norm = gamma_st[i];
      idxGammaMax = i; 
    }
  }
  value /= gamma_st[idxGammaMax];

  return value;
}

} // end nalu namespace
} // end sierra namespace

