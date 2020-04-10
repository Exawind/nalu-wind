#include "matrix_free/Coefficients.h"

#include <cmath>

#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

constexpr Coeffs<1>::nodal_matrix_type Coeffs<1>::W;
constexpr Coeffs<1>::nodal_matrix_type Coeffs<1>::D;
constexpr Coeffs<1>::scs_matrix_type Coeffs<1>::Nt;
constexpr Coeffs<1>::scs_matrix_type Coeffs<1>::Dt;
constexpr Coeffs<1>::linear_nodal_matrix_type Coeffs<1>::Nlin;
constexpr Coeffs<1>::linear_scs_matrix_type Coeffs<1>::Ntlin;

constexpr Coeffs<2>::nodal_matrix_type Coeffs<2>::W;
constexpr Coeffs<2>::nodal_matrix_type Coeffs<2>::D;
constexpr Coeffs<2>::scs_matrix_type Coeffs<2>::Nt;
constexpr Coeffs<2>::scs_matrix_type Coeffs<2>::Dt;
constexpr Coeffs<2>::linear_nodal_matrix_type Coeffs<2>::Nlin;
constexpr Coeffs<2>::linear_scs_matrix_type Coeffs<2>::Ntlin;

constexpr Coeffs<3>::nodal_matrix_type Coeffs<3>::W;
constexpr Coeffs<3>::nodal_matrix_type Coeffs<3>::D;
constexpr Coeffs<3>::scs_matrix_type Coeffs<3>::Nt;
constexpr Coeffs<3>::scs_matrix_type Coeffs<3>::Dt;
constexpr Coeffs<3>::linear_nodal_matrix_type Coeffs<3>::Nlin;
constexpr Coeffs<3>::linear_scs_matrix_type Coeffs<3>::Ntlin;

constexpr Coeffs<4>::nodal_matrix_type Coeffs<4>::W;
constexpr Coeffs<4>::nodal_matrix_type Coeffs<4>::D;
constexpr Coeffs<4>::scs_matrix_type Coeffs<4>::Nt;
constexpr Coeffs<4>::scs_matrix_type Coeffs<4>::Dt;
constexpr Coeffs<4>::linear_nodal_matrix_type Coeffs<4>::Nlin;

constexpr LocalArray<double[2]> Coeffs<1>::Wl;
constexpr LocalArray<double[3]> Coeffs<2>::Wl;
constexpr LocalArray<double[4]> Coeffs<3>::Wl;
constexpr LocalArray<double[5]> Coeffs<4>::Wl;

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
