#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

constexpr double kEps = 1e-10;
constexpr double kTwoPi = 2.0 * M_PI;

using Vec = std::array<double, 3>;
using Mat = std::array<double, 9>;
using Joints = std::array<double, 7>;

inline Vec add(Vec a, Vec b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec sub(Vec a, Vec b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec scale(Vec a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(Vec a, Vec b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec cross(Vec a, Vec b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(Vec a) { return std::sqrt(dot(a, a)); }
inline bool finite(Vec value) {
  return std::all_of(value.begin(), value.end(), [](double item) { return std::isfinite(item); });
}
inline bool finite(Mat value) {
  return std::all_of(value.begin(), value.end(), [](double item) { return std::isfinite(item); });
}
inline Vec unit(Vec a) {
  const double n = norm(a);
  if (n < kEps) throw std::runtime_error("Cannot normalize a near-zero vector");
  return scale(a, 1.0 / n);
}
inline Mat eye() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
inline Mat transpose(Mat a) { return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}; }
inline Vec mul(Mat a, Vec v) {
  return {a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
          a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
}
inline Mat mul(Mat a, Mat b) {
  Mat c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return c;
}
inline Mat rot(Vec axis, double theta) {
  axis = unit(axis);
  const double c = std::cos(theta), s = std::sin(theta), d = 1.0 - c;
  const double x = axis[0], y = axis[1], z = axis[2];
  return {c + x * x * d,     x * y * d - z * s, x * z * d + y * s, y * x * d + z * s, c + y * y * d,
          y * z * d - x * s, z * x * d - y * s, z * y * d + x * s, c + z * z * d};
}

inline double determinant(Mat matrix) {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
         matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
         matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

inline bool rotation_matrix(Mat matrix, double tolerance = 1e-7) {
  if (!finite(matrix) || std::abs(determinant(matrix) - 1.0) > tolerance) return false;
  const Mat product = mul(transpose(matrix), matrix);
  const Mat identity = eye();
  double error_squared = 0.0;
  for (size_t i = 0; i < product.size(); ++i)
    error_squared += (product[i] - identity[i]) * (product[i] - identity[i]);
  return std::sqrt(error_squared) <= tolerance;
}
