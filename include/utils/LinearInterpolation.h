#ifndef LINEARINTERPOLATION_H
#define LINEARINTERPOLATION_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <utility>

#if defined(__GNUC__) && (6 < __GNUC__)
#define NALU_CASE_FALLTHROUGH __attribute__ ((fallthrough));
#else
#define NALU_CASE_FALLTHROUGH
#endif

namespace sierra {
namespace nalu {
namespace utils {

struct OutOfBounds
{
  //! Out of bounds limit types
  enum boundLimits {
    LOWLIM = -2, //!< xtgt < xarray[0]
    UPLIM = -1,  //!< xtgt > xarray[N]
    VALID = 0    //!< xarray[0] <= xtgt <= xarray[N]
  };

  //! Flags indicating action to perform on Out of Bounds situation
  enum OobAction {
    ERROR = 0,  //!< Raise runtime error
    WARN,       //!< Warn and then CLAMP
    CLAMP,      //!< Clamp values to the end points
    EXTRAPOLATE //!< Extrapolate linearly based on end point
  };
};

template <typename T>
using Array1D = std::vector<T>;

template <typename T>
struct InterpTraits
{
  typedef typename Array1D<T>::size_type size_type;
  typedef typename std::pair<OutOfBounds::boundLimits, size_type> index_type;
};

/**
 * Determine whether the given value is within the limits of the interpolation
 * table
 */
template <typename T>
inline typename InterpTraits<T>::index_type
check_bounds(const Array1D<T>& xinp, const T& x)
{
  auto sz = xinp.size();

  if (sz < 2) {
    throw std::runtime_error(
      "Interpolation table contains less than 2 entries.");
  }

  if (x < xinp[0]) {
    return std::make_pair(OutOfBounds::LOWLIM, 0);
  } else if (x > xinp[sz - 1]) {
    return std::make_pair(OutOfBounds::UPLIM, sz - 1);
  } else {
    return std::make_pair(OutOfBounds::VALID, 0);
  }
}

/**
 * Return an index object corresponding to the x-value based on interpolation
 * table.
 *
 * The `std::pair` returned contains two values: the bounds indicator and the
 * index of the element in the interpolation table such that `xarray[i] <= x <
 * xarray[i+1]`
 */
template <typename T>
inline typename InterpTraits<T>::index_type
find_index(const Array1D<T>& xinp, const T& x)
{
  auto idx = check_bounds(xinp, x);
  if (idx.first == OutOfBounds::UPLIM || idx.first == OutOfBounds::LOWLIM)
    return idx;

  auto sz = xinp.size();
  for (size_t i = 1; i < sz; i++) {
    if (x <= xinp[i]) {
      idx.second = i - 1;
      break;
    }
  }
  return idx;
}

/**
 * Perform a 1-D linear interpolation
 */
template <typename T>
void
linear_interp(
  const Array1D<T>& xinp,
  const Array1D<T>& yinp,
  const T& xout,
  T& yout,
  OutOfBounds::OobAction oob = OutOfBounds::CLAMP)
{
  auto idx = find_index(xinp, xout);

  switch (idx.first) {
  case OutOfBounds::LOWLIM:
  case OutOfBounds::UPLIM: {
    switch (oob) {
    case OutOfBounds::ERROR:
      throw std::runtime_error("Out of bounds error in interpolation");
      break;

    case OutOfBounds::WARN:
      std::cout
        << std::endl
        << "WARNING: Out of bound values encountered during interpolation"
        << std::endl;
    // no break here... allow fallthrough
        NALU_CASE_FALLTHROUGH   
    case OutOfBounds::CLAMP:
      yout = yinp[idx.second];
      break;

    case OutOfBounds::EXTRAPOLATE: {
      auto ii = idx.second;
      if ((ii + 1) == xinp.size())
        --ii;

      yout = yinp[ii] +
             (yinp[ii + 1] - yinp[ii]) / (xinp[ii + 1] - xinp[ii]) *
               (xout - xinp[ii]);
      break;
    }
    }
    break;
  }
  case OutOfBounds::VALID:
    auto j = idx.second;
    T fac = (xout - xinp[j]) / (xinp[j + 1] - xinp[j]);
    yout = (static_cast<T>(1.0) - fac) * yinp[j] + fac * yinp[j + 1];
    break;
  }
}

} // namespace utils
} // namespace nalu
} // namespace sierra

#endif /* LINEARINTERPOLATION_H */
