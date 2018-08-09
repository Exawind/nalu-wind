/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperatorsHexInternal_h
#define HighOrderOperatorsHexInternal_h

#include <CVFEMTypeDefs.h>
#include <master_element/DirectionMacros.h>

#include <cmath>

namespace sierra {
namespace nalu {


// Tensor contractions for SIMD types
// maybe specialize to switch to BLAS if value type is float/double

namespace tensor_internal {

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_x(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_scalar_view<p, Scalar>& in,
  nodal_scalar_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        Scalar acc = 0;
        for (int q = 0; q < n; ++q) {
          acc += coeffMatrix(i, q) * in(k, j, q);
        }
        out(k, j, i) = acc;
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_x(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_scalar_view<p, Scalar>& in,
  nodal_vector_view<p, Scalar>& out, int component)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        Scalar acc = 0;
        for (int q = 0; q < n; ++q) {
          acc += coeffMatrix(i, q) * in(k, j, q);
        }
        out(k, j, i, component) = acc;
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_x(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_vector_view<p, Scalar>& in,
  nodal_vector_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        Scalar accx = 0;
        Scalar accy = 0;
        Scalar accz = 0;
        for (int q = 0; q < n; ++q) {
          const Scalar coeff = coeffMatrix(i, q);
          accx += coeff * in(k, j, q, XH);
          accy += coeff * in(k, j, q, YH);
          accz += coeff * in(k, j, q, ZH);
        }
        out(k, j, i, XH) = accx;
        out(k, j, i, YH) = accy;
        out(k, j, i, ZH) = accz;
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_x(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_vector_view<p, Scalar>& in,
  nodal_tensor_view<p, Scalar>& out, int out_comp)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        Scalar accx = 0;
        Scalar accy = 0;
        Scalar accz = 0;
        for (int q = 0; q < n; ++q) {
          const auto coeff = coeffMatrix(i, q);
          accx += coeff * in(k, j, q, XH);
          accy += coeff * in(k, j, q, YH);
          accz += coeff * in(k, j, q, ZH);
        }
        out(k, j, i, XH, out_comp) = accx;
        out(k, j, i, YH, out_comp) = accy;
        out(k, j, i, ZH, out_comp) = accz;
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_y(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_scalar_view<p, Scalar>& in,
  nodal_scalar_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        out(k, j, i) = 0;
      }
      for (int q = 0; q < n; ++q) {
        const auto temp = coeffMatrix(j, q);
        for (int i = 0; i < nx; ++i) {
          out(k, j, i) += temp * in(k, q, i);
        }
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_y(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_scalar_view<p, Scalar>& in,
  nodal_vector_view<p, Scalar>& out, int component)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        out(k, j, i, component) = 0;
      }
      for (int q = 0; q < n; ++q) {
        const auto temp = coeffMatrix(j, q);
        for (int i = 0; i < nx; ++i) {
          out(k, j, i, component) += temp * in(k, q, i);
        }
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_y(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_vector_view<p, Scalar>& in,
  nodal_vector_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        out(k, j, i, XH) = 0.0;
        out(k, j, i, YH) = 0.0;
        out(k, j, i, ZH) = 0.0;
      }

      for (int q = 0; q < n; ++q) {
        const auto temp = coeffMatrix(j, q);
        for (int i = 0; i < nx; ++i) {
          out(k, j, i, XH) += temp * in(k, q, i, XH);
          out(k, j, i, YH) += temp * in(k, q, i, YH);
          out(k, j, i, ZH) += temp * in(k, q, i, ZH);
        }
      }
    }
  }
}
template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_y(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_vector_view<p, Scalar>& in,
  nodal_tensor_view<p, Scalar>& out, int out_col)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        Scalar accx = 0;
        Scalar accy = 0;
        Scalar accz = 0;
        for (int q = 0; q < n; ++q) {
          const auto temp = coeffMatrix(j, q);
          accx += temp * in(k, q, i, XH);
          accy += temp * in(k, q, i, YH);
          accz += temp * in(k, q, i, ZH);
        }
        out(k, j, i, XH, out_col) = accx;
        out(k, j, i, YH, out_col) = accy;
        out(k, j, i, ZH, out_col) = accz;
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_z(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_scalar_view<p, Scalar>& in,
  nodal_scalar_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        out(k, j, i) = 0;
      }
    }

    for (int q = 0; q < n; ++q) {
      const auto temp = coeffMatrix(k, q);
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          out(k, j, i) += temp * in(q, j, i);
        }
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_z(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_scalar_view<p, Scalar>& in,
  nodal_vector_view<p, Scalar>& out, int component)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        out(k, j, i, component) = 0;
      }
    }

    for (int q = 0; q < n; ++q) {
      const auto temp = coeffMatrix(k, q);
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          out(k, j, i, component) += temp * in(q, j, i);
        }
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_z(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_vector_view<p, Scalar>& in,
  nodal_vector_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        out(k, j, i, XH) = 0.0;
        out(k, j, i, YH) = 0.0;
        out(k, j, i, ZH) = 0.0;
      }
    }

    for (int q = 0; q < n; ++q) {
      const auto temp = coeffMatrix(k, q);
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          out(k, j, i, XH) += temp * in(q, j, i, XH);
          out(k, j, i, YH) += temp * in(q, j, i, YH);
          out(k, j, i, ZH) += temp * in(q, j, i, ZH);
        }
      }
    }
  }
}

template<int p, typename Scalar, int nx, int ny, int nz>
KOKKOS_FORCEINLINE_FUNCTION
void apply_z(const scs_matrix_view<p, Scalar>& coeffMatrix, const nodal_vector_view<p, Scalar>& in,
  nodal_tensor_view<p, Scalar>& out, int out_col)
{
  constexpr int n = p + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        Scalar accx = 0;
        Scalar accy = 0;
        Scalar accz = 0;
        for (int q = 0; q < n; ++q) {
          const auto temp = coeffMatrix(k, q);
          accx += temp * in(q, j, i, XH);
          accy += temp * in(q, j, i, YH);
          accz += temp * in(q, j, i, ZH);
        }
        out(k, j, i, XH, out_col) = accx;
        out(k, j, i, YH, out_col) = accy;
        out(k, j, i, ZH, out_col) = accz;
      }
    }
  }
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void difference_x(const nodal_scalar_view<p, Scalar>& in, nodal_scalar_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      out(k, j, 0) -= in(k, j, 0);
      for (int q = 1; q < p; ++q) {
        out(k, j, q) -= in(k, j, q) - in(k, j, q - 1);
      }
      out(k, j, p) += in(k, j, p - 1);
    }
  }
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void difference_x(const nodal_vector_view<p, Scalar>& in, nodal_vector_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      out(k, j, 0, XH) -= in(k, j, 0, XH);
      out(k, j, 0, YH) -= in(k, j, 0, YH);
      out(k, j, 0, ZH) -= in(k, j, 0, ZH);
      for (int q = 1; q < p; ++q) {
        out(k, j, q, XH) -= in(k, j, q, XH) - in(k, j, q - 1, XH);
        out(k, j, q, YH) -= in(k, j, q, YH) - in(k, j, q - 1, YH);
        out(k, j, q, ZH) -= in(k, j, q, ZH) - in(k, j, q - 1, ZH);

      }
      out(k, j, p, XH) += in(k, j, p - 1, XH);
      out(k, j, p, YH) += in(k, j, p - 1, YH);
      out(k, j, p, ZH) += in(k, j, p - 1, ZH);
    }
  }
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void difference_y(const nodal_scalar_view<p, Scalar>& in, nodal_scalar_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
      out(k, 0, i) -= in(k, 0, i);
      for (int q = 1; q < p; ++q) {
        out(k, q, i) -= in(k, q, i) - in(k, q - 1, i);
      }
      out(k, p, i) += in(k, p - 1, i);
    }
  }
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void difference_y(const nodal_vector_view<p, Scalar>& in, nodal_vector_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
      out(k, 0, i, XH) -= in(k, 0, i, XH);
      out(k, 0, i, YH) -= in(k, 0, i, YH);
      out(k, 0, i, ZH) -= in(k, 0, i, ZH);
      for (int q = 1; q < p; ++q) {
        out(k, q, i, XH) -= in(k, q, i, XH) - in(k, q - 1, i, XH);
        out(k, q, i, YH) -= in(k, q, i, YH) - in(k, q - 1, i, YH);
        out(k, q, i, ZH) -= in(k, q, i, ZH) - in(k, q - 1, i, ZH);
      }
      out(k, p, i, XH) += in(k, p - 1, i, XH);
      out(k, p, i, YH) += in(k, p - 1, i, YH);
      out(k, p, i, ZH) += in(k, p - 1, i, ZH);
    }
  }
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void difference_z(const nodal_scalar_view<p, Scalar>& in, nodal_scalar_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(0, j, i) -= in(0, j, i);
      for (int q = 1; q < p; ++q) {
        out(q, j, i) -= in(q, j, i) - in(q - 1, j, i);
      }
      out(p, j, i) += in(p - 1, j, i);
    }
  }
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void difference_z(const nodal_vector_view<p, Scalar>& in, nodal_vector_view<p, Scalar>& out)
{
  constexpr int n = p + 1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(0, j, i, XH) -= in(0, j, i, XH);
      out(0, j, i, YH) -= in(0, j, i, YH);
      out(0, j, i, ZH) -= in(0, j, i, ZH);
      for (int q = 1; q < p; ++q) {
        out(q, j, i, XH) -= in(q, j, i, XH) - in(q - 1, j, i, XH);
        out(q, j, i, YH) -= in(q, j, i, YH) - in(q - 1, j, i, YH);
        out(q, j, i, ZH) -= in(q, j, i, ZH) - in(q - 1, j, i, ZH);
      }
      out(p, j, i, XH) += in(p - 1, j, i, XH);
      out(p, j, i, YH) += in(p - 1, j, i, YH);
      out(p, j, i, ZH) += in(p - 1, j, i, ZH);
    }
  }
}

}

} // namespace Sierra
}

#endif

