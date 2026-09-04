 #include <Kokkos_Core.hpp>

namespace sierra {
namespace kynema_ugf {

/**
 * @brief Rotates provided vector by provided *unit* quaternion and returns the
 * result
 */
template <typename Quaternion, typename View1, typename View2>
KOKKOS_INLINE_FUNCTION void
RotateVectorByQuaternion(const Quaternion& q, const View1& v, View2& v_rot)
{
  v_rot[0] = (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]) * v[0] +
             2. * (q[1] * q[2] - q[0] * q[3]) * v[1] +
             2. * (q[1] * q[3] + q[0] * q[2]) * v[2];
  v_rot[1] = 2. * (q[1] * q[2] + q[0] * q[3]) * v[0] +
             (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]) * v[1] +
             2. * (q[2] * q[3] - q[0] * q[1]) * v[2];
  v_rot[2] = 2. * (q[1] * q[3] - q[0] * q[2]) * v[0] +
             2. * (q[2] * q[3] + q[0] * q[1]) * v[1] +
             (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]) * v[2];
}

/**
 * @brief Rotates provided vector by inverse of provided *unit* quaternion and
 * returns the result
 */
template <typename Quaternion, typename View1, typename View2>
KOKKOS_INLINE_FUNCTION void
RotateVectorByQuaternionInv(const Quaternion& q, const View1& v, View2& v_rot)
{
  v_rot[0] = (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]) * v[0] +
             2. * (q[1] * q[2] + q[0] * q[3]) * v[1] +
             2. * (q[1] * q[3] - q[0] * q[2]) * v[2];
  v_rot[1] = 2. * (q[1] * q[2] - q[0] * q[3]) * v[0] +
             (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]) * v[1] +
             2. * (q[2] * q[3] + q[0] * q[1]) * v[2];
  v_rot[2] = 2. * (q[1] * q[3] + q[0] * q[2]) * v[0] +
             2. * (q[2] * q[3] - q[0] * q[1]) * v[1] +
             (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]) * v[2];
}

/**
 * @brief Computes 3D cross product of two vectors and stores the result.
 */
template <typename View1, typename View2, typename View3>
KOKKOS_INLINE_FUNCTION void
CrossProduct3(const View1& a, const View2& b, View3& c)
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * @brief Computes barycentric weights for interpolation nodes.
 *
 * Compute once for a given node set and reuse for many evaluations.
 * This is O(n^2) and intended for setup, not the hot loop.
 *
 * @param xs Interpolation nodes
 * @param n Number of interpolation nodes
 * @param bary_weights Output barycentric weights (size n)
 */
KOKKOS_INLINE_FUNCTION
inline void
ComputeBarycentricWeights(
  const double* KOKKOS_RESTRICT xs, int n, double* KOKKOS_RESTRICT bary_weights)
{
  for (int j = 0; j < n; ++j) {
    double w = 1.0;
    for (int m = 0; m < n; ++m) {
      if (j != m) {
        w /= (xs[j] - xs[m]);
      }
    }
    bary_weights[j] = w;
  }
}

/**
 * @brief Computes Lagrange interpolation basis weights at x.
 *
 * Barycentric form is O(n) per evaluation and avoids dynamic allocations,
 * making it suitable for use inside Kokkos kernels.
 *
 * @param x Evaluation point
 * @param xs Interpolation nodes
 * @param bary_weights Precomputed barycentric node weights
 * @param n Number of interpolation nodes
 * @param weights Output interpolation basis weights (size n)
 */
KOKKOS_INLINE_FUNCTION
inline void
LagrangePolynomialInterpWeights(
  double x,
  const double* KOKKOS_RESTRICT xs,
  const double* KOKKOS_RESTRICT bary_weights,
  int n,
  double* KOKKOS_RESTRICT weights)
{
  constexpr double tol = 1.e-14;

  // If x coincides with a node, return exact Kronecker-delta basis values.
  for (int j = 0; j < n; ++j) {
    if (std::abs(x - xs[j]) <= tol) {
      for (int k = 0; k < n; ++k) {
        weights[k] = 0.0;
      }
      weights[j] = 1.0;
      return;
    }
  }

  double denom = 0.0;
  for (int j = 0; j < n; ++j) {
    const double tmp = bary_weights[j] / (x - xs[j]);
    weights[j] = tmp;
    denom += tmp;
  }

  const double inv_denom = 1.0 / denom;
  for (int j = 0; j < n; ++j) {
    weights[j] *= inv_denom;
  }
}

/**
 * @brief Computes Lagrange basis values and their xi-derivatives in one pass.
 *
 * At interior points uses the barycentric derivative formula:
 *   dl_j/dx = l_j * (S2/S - 1/(x - x_j))
 * where S = sum_k w_k/(x-x_k) and S2 = sum_k w_k/(x-x_k)^2.
 * At exact nodes uses the standard nodal derivative formula.
 *
 * @param x Evaluation point
 * @param xs Interpolation nodes
 * @param bary_weights Precomputed barycentric node weights
 * @param n Number of interpolation nodes
 * @param weights Output basis values l_j(x) (size n)
 * @param dweights Output basis derivatives dl_j/dx (size n)
 */
KOKKOS_INLINE_FUNCTION
inline void
LagrangePolynomialInterpWeightsAndDerivatives(
  double x,
  const double* KOKKOS_RESTRICT xs,
  const double* KOKKOS_RESTRICT bary_weights,
  int n,
  double* KOKKOS_RESTRICT weights,
  double* KOKKOS_RESTRICT dweights)
{
  constexpr double tol = 1.e-14;

  // Exact-node case: l_m=1, l_j=0 for j!=m; use standard nodal derivative
  // formula.
  for (int j = 0; j < n; ++j) {
    if (std::abs(x - xs[j]) <= tol) {
      for (int k = 0; k < n; ++k) {
        weights[k] = 0.0;
        dweights[k] = 0.0;
      }
      weights[j] = 1.0;
      double sum_d = 0.0;
      for (int k = 0; k < n; ++k) {
        if (k != j) {
          dweights[k] = (bary_weights[k] / bary_weights[j]) / (xs[j] - xs[k]);
          sum_d += dweights[k];
        }
      }
      dweights[j] = -sum_d;
      return;
    }
  }

  // General case: single-pass over nodes accumulates S and S2.
  // S  = sum_k w_k/(x-x_k)   (barycentric denominator)
  // S2 = sum_k w_k/(x-x_k)^2 (needed for derivative)
  double S = 0.0;
  double S2 = 0.0;
  for (int j = 0; j < n; ++j) {
    const double diff = x - xs[j];
    const double t = bary_weights[j] / diff;
    weights[j] = t; // temporarily store t_j = w_j/(x-x_j)
    S += t;
    S2 += t / diff;
  }

  const double inv_S = 1.0 / S;
  const double ratio = S2 * inv_S; // S2/S, scalar shared across all j
  for (int j = 0; j < n; ++j) {
    const double lj = weights[j] * inv_S;
    weights[j] = lj;
    dweights[j] = lj * (ratio - 1.0 / (x - xs[j]));
  }
}

/**
 * @brief Computes shape function matrix ϕg relating points ξb to ξg.
 *
 * Layout of shape_functions is row-major with [output][input], flattened as
 * shape_functions[output_idx * num_input_points + input_idx].
 *
 * @param input_points Input points ξb in [-1, 1]
 * @param num_input_points Number of input points
 * @param output_points Output points ξg in [-1, 1]
 * @param output_bary_weights Precomputed barycentric weights for output_points
 * @param num_output_points Number of output points
 * @param scratch_weights Scratch array of size num_output_points
 * @param shape_functions Output flattened shape matrix [num_output_points *
 * num_input_points]
 */
KOKKOS_INLINE_FUNCTION
inline void
ComputeShapeFunctionValues(
  const double* KOKKOS_RESTRICT input_points,
  int num_input_points,
  const double* KOKKOS_RESTRICT output_points,
  const double* KOKKOS_RESTRICT output_bary_weights,
  int num_output_points,
  double* KOKKOS_RESTRICT scratch_weights,
  double* KOKKOS_RESTRICT shape_functions)
{
  for (int input_point = 0; input_point < num_input_points; ++input_point) {
    LagrangePolynomialInterpWeights(
      input_points[input_point], output_points, output_bary_weights,
      num_output_points, scratch_weights);

    for (int output_point = 0; output_point < num_output_points;
         ++output_point) {
      shape_functions[output_point * num_input_points + input_point] =
        scratch_weights[output_point];
    }
  }
}

/**
 * @brief Interpolates a field at an evaluation point using Lagrange basis
 * functions.
 *
 * Computes the interpolated value of a field at a given parametric coordinate
 * using precomputed barycentric weights and Lagrange basis functions.
 *
 * @param x_eval Evaluation point in parameter space
 * @param node_xi Node parametric coordinates (size num_nodes)
 * @param node_field Node field values, layout [num_nodes][field_stride]
 * @param num_nodes Number of nodes
 * @param field_stride Number of components per node (e.g., 7 for position, 3
 * for displacement)
 * @param bary_weights Precomputed barycentric weights for node_xi (size
 * num_nodes)
 * @param scratch_weights Scratch array for basis functions (size num_nodes)
 * @param interp_field Output interpolated field (size field_stride)
 */
KOKKOS_INLINE_FUNCTION
inline void
InterpolateFieldAtPoint(
  double x_eval,
  const double* KOKKOS_RESTRICT node_xi,
  const double* KOKKOS_RESTRICT node_field,
  int num_nodes,
  int field_stride,
  const double* KOKKOS_RESTRICT bary_weights,
  double* KOKKOS_RESTRICT scratch_weights,
  double* KOKKOS_RESTRICT interp_field)
{
  // Compute Lagrange basis weights at x_eval
  LagrangePolynomialInterpWeights(
    x_eval, node_xi, bary_weights, num_nodes, scratch_weights);

  // Initialize output to zero
  for (int c = 0; c < field_stride; ++c) {
    interp_field[c] = 0.0;
  }

  // Interpolate each component of the field
  for (int j = 0; j < num_nodes; ++j) {
    const double basis_j = scratch_weights[j];
    for (int c = 0; c < field_stride; ++c) {
      interp_field[c] += basis_j * node_field[j * field_stride + c];
    }
  }
}

KOKKOS_INLINE_FUNCTION
inline double
SquaredDistance3(
  const double* KOKKOS_RESTRICT a, const double* KOKKOS_RESTRICT b)
{
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

KOKKOS_INLINE_FUNCTION
inline double
BladePointDistanceSquaredAtXi(
  double xi,
  const double* KOKKOS_RESTRICT query_point,
  const double* KOKKOS_RESTRICT node_xi,
  const double* KOKKOS_RESTRICT node_positions,
  int num_nodes,
  const double* KOKKOS_RESTRICT bary_weights,
  double* KOKKOS_RESTRICT scratch_weights,
  double* KOKKOS_RESTRICT interp_position)
{
  // Interpolate only xyz from node_positions[num_nodes][7] to reduce work in
  // the hot path.
  LagrangePolynomialInterpWeights(
    xi, node_xi, bary_weights, num_nodes, scratch_weights);

  interp_position[0] = 0.0;
  interp_position[1] = 0.0;
  interp_position[2] = 0.0;
  for (int j = 0; j < num_nodes; ++j) {
    const double basis_j = scratch_weights[j];
    interp_position[0] += basis_j * node_positions[j * 7 + 0];
    interp_position[1] += basis_j * node_positions[j * 7 + 1];
    interp_position[2] += basis_j * node_positions[j * 7 + 2];
  }

  return SquaredDistance3(interp_position, query_point);
}

/**
 * @brief Evaluates f'(xi) = 2*(r(xi)-q) . r'(xi) in one pass.
 *
 * r(xi) and r'(xi) are the interpolated blade position and its xi-derivative.
 * f'(xi) = 0 is the necessary condition for a closest-point projection.
 *
 * @param xi Parametric evaluation point
 * @param query_point Target point [x, y, z]
 * @param node_xi Blade nodal xi coordinates
 * @param node_positions Blade nodal positions, flattened [num_nodes][7]
 * @param num_nodes Number of blade nodes
 * @param bary_weights Precomputed barycentric weights for node_xi
 * @param scratch_weights Scratch array size num_nodes (basis values)
 * @param scratch_dweights Scratch array size num_nodes (basis derivatives)
 */
KOKKOS_INLINE_FUNCTION
inline double
BladePointFPrimeAtXi(
  double xi,
  const double* KOKKOS_RESTRICT query_point,
  const double* KOKKOS_RESTRICT node_xi,
  const double* KOKKOS_RESTRICT node_positions,
  int num_nodes,
  const double* KOKKOS_RESTRICT bary_weights,
  double* KOKKOS_RESTRICT scratch_weights,
  double* KOKKOS_RESTRICT scratch_dweights)
{
  LagrangePolynomialInterpWeightsAndDerivatives(
    xi, node_xi, bary_weights, num_nodes, scratch_weights, scratch_dweights);

  double r[3] = {0.0, 0.0, 0.0};
  double rprime[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < num_nodes; ++j) {
    const double lj = scratch_weights[j];
    const double dlj = scratch_dweights[j];
    r[0] += lj * node_positions[j * 7 + 0];
    r[1] += lj * node_positions[j * 7 + 1];
    r[2] += lj * node_positions[j * 7 + 2];
    rprime[0] += dlj * node_positions[j * 7 + 0];
    rprime[1] += dlj * node_positions[j * 7 + 1];
    rprime[2] += dlj * node_positions[j * 7 + 2];
  }

  return 2.0 * ((r[0] - query_point[0]) * rprime[0] +
                (r[1] - query_point[1]) * rprime[1] +
                (r[2] - query_point[2]) * rprime[2]);
}

/**
 * @brief Finds closest point on blade centerline to a query point.
 *
 * Minimizes squared distance over xi in [-1, 1] with:
 * 1) coarse uniform scan to bracket the minimum
 * 2) bracketed secant method on f'(xi)=0 for fast refinement
 *
 * @param query_point Target point [x, y, z]
 * @param node_xi Blade nodal xi coordinates
 * @param node_positions Blade nodal positions, flattened [num_nodes][7]
 * @param num_nodes Number of blade nodes
 * @param bary_weights Barycentric weights for node_xi
 * @param scratch_weights Scratch array size num_nodes (basis values)
 * @param scratch_dweights Scratch array size num_nodes (basis derivatives)
 * @param out_xi Closest xi in [-1, 1]
 * @param out_position Closest interpolated position [x, y, z]
 * @param out_dist2 Minimum squared distance
 */
KOKKOS_INLINE_FUNCTION
inline void
FindClosestPointOnBlade(
  const double* KOKKOS_RESTRICT query_point,
  const double* KOKKOS_RESTRICT node_xi,
  const double* KOKKOS_RESTRICT node_positions,
  int num_nodes,
  const double* KOKKOS_RESTRICT bary_weights,
  double* KOKKOS_RESTRICT scratch_weights,
  double* KOKKOS_RESTRICT scratch_dweights,
  double& out_xi,
  double* KOKKOS_RESTRICT out_position,
  double& out_dist2)
{
  constexpr int kCoarseSamples = 10;
  constexpr int kSecantIters = 8;
  constexpr double kXiMin = -1.0;
  constexpr double kXiMax = 1.0;

  double interp_tmp[3];

  // Stage 1: coarse uniform scan to locate the bracket containing the minimum.
  int best_index = 0;
  double best_xi = kXiMin;
  double best_dist2 = BladePointDistanceSquaredAtXi(
    best_xi, query_point, node_xi, node_positions, num_nodes, bary_weights,
    scratch_weights, interp_tmp);

  for (int i = 1; i < kCoarseSamples; ++i) {
    const double t =
      static_cast<double>(i) / static_cast<double>(kCoarseSamples - 1);
    const double xi = kXiMin + (kXiMax - kXiMin) * t;
    const double d2 = BladePointDistanceSquaredAtXi(
      xi, query_point, node_xi, node_positions, num_nodes, bary_weights,
      scratch_weights, interp_tmp);
    if (d2 < best_dist2) {
      best_dist2 = d2;
      best_xi = xi;
      best_index = i;
    }
  }

  const int left_index = (best_index > 0) ? (best_index - 1) : 0;
  const int right_index =
    (best_index < kCoarseSamples - 1) ? (best_index + 1) : (kCoarseSamples - 1);

  double a = kXiMin + (kXiMax - kXiMin) * static_cast<double>(left_index) /
                        static_cast<double>(kCoarseSamples - 1);
  double b = kXiMin + (kXiMax - kXiMin) * static_cast<double>(right_index) /
                        static_cast<double>(kCoarseSamples - 1);

  // Stage 2: bracketed secant on f'(xi) = 2*(r(xi)-q).r'(xi) = 0.
  // The bracket [a, b] straddles the minimum so f'(a) <= 0 <= f'(b).
  double fprime_a = BladePointFPrimeAtXi(
    a, query_point, node_xi, node_positions, num_nodes, bary_weights,
    scratch_weights, scratch_dweights);
  double fprime_b = BladePointFPrimeAtXi(
    b, query_point, node_xi, node_positions, num_nodes, bary_weights,
    scratch_weights, scratch_dweights);

  if (fprime_a * fprime_b >= 0.0) {
    // Bracket has no sign change: minimum is at or very near a boundary.
    // Fall back to the coarse-scan best.
    out_xi = best_xi;
  } else {
    for (int iter = 0; iter < kSecantIters; ++iter) {
      // Secant step: linear interpolation of f' to find its zero.
      const double denom = fprime_b - fprime_a;
      const double xi_sec = (std::abs(denom) > 1.e-30)
                              ? (a * fprime_b - b * fprime_a) / denom
                              : 0.5 * (a + b);
      // Clamp to bracket as a safeguard.
      const double xi_new = Kokkos::min(b, Kokkos::max(a, xi_sec));
      const double fprime_new = BladePointFPrimeAtXi(
        xi_new, query_point, node_xi, node_positions, num_nodes, bary_weights,
        scratch_weights, scratch_dweights);
      if (std::abs(fprime_new) < 1.e-14) {
        a = xi_new;
        b = xi_new;
        break;
      }
      // Narrow the bracket.
      if (fprime_new * fprime_a < 0.0) {
        b = xi_new;
        fprime_b = fprime_new;
      } else {
        a = xi_new;
        fprime_a = fprime_new;
      }
      if (b - a < 1.e-12)
        break;
    }
    out_xi = 0.5 * (a + b);
  }

  // Clamp to valid domain.
  out_xi = Kokkos::min(kXiMax, Kokkos::max(kXiMin, out_xi));

  // Final position and distance at the converged xi.
  out_dist2 = BladePointDistanceSquaredAtXi(
    out_xi, query_point, node_xi, node_positions, num_nodes, bary_weights,
    scratch_weights, interp_tmp);
  out_position[0] = interp_tmp[0];
  out_position[1] = interp_tmp[1];
  out_position[2] = interp_tmp[2];
}

} // namespace kynema_ugf
} // namespace sierra