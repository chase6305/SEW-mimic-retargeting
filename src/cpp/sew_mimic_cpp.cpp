// Native C++17 implementation of the SEW-Mimic parallel-wrist solver.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;
constexpr double kEps = 1e-10;
constexpr double kTwoPi = 2.0 * M_PI;

using Vec = std::array<double, 3>;
using Mat = std::array<double, 9>;
using Joints = std::array<double, 7>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

Vec add(Vec a, Vec b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec sub(Vec a, Vec b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec scale(Vec a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
double dot(Vec a, Vec b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(Vec a, Vec b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(Vec a) { return std::sqrt(dot(a, a)); }
bool finite(Vec value) {
  return std::all_of(value.begin(), value.end(), [](double item) { return std::isfinite(item); });
}
bool finite(Mat value) {
  return std::all_of(value.begin(), value.end(), [](double item) { return std::isfinite(item); });
}
Vec unit(Vec a) {
  const double n = norm(a);
  if (n < kEps) throw std::runtime_error("Cannot normalize a near-zero vector");
  return scale(a, 1.0 / n);
}
Mat eye() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
Mat transpose(Mat a) { return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}; }
Vec mul(Mat a, Vec v) {
  return {a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
          a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
}
Mat mul(Mat a, Mat b) {
  Mat c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return c;
}
Mat rot(Vec axis, double theta) {
  axis = unit(axis);
  const double c = std::cos(theta), s = std::sin(theta), d = 1.0 - c;
  const double x = axis[0], y = axis[1], z = axis[2];
  return {c + x * x * d,     x * y * d - z * s, x * z * d + y * s, y * x * d + z * s, c + y * y * d,
          y * z * d - x * s, z * x * d - y * s, z * y * d + x * s, c + z * z * d};
}

double determinant(Mat matrix) {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
         matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
         matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

bool rotation_matrix(Mat matrix, double tolerance = 1e-7) {
  if (!finite(matrix) || std::abs(determinant(matrix) - 1.0) > tolerance) return false;
  const Mat product = mul(transpose(matrix), matrix);
  const Mat identity = eye();
  double error_squared = 0.0;
  for (size_t i = 0; i < product.size(); ++i)
    error_squared += (product[i] - identity[i]) * (product[i] - identity[i]);
  return std::sqrt(error_squared) <= tolerance;
}

/** Calibrated orientation-only model shared by every native solve.
 *
 * Matrices follow the same row-major parent<-child convention as Serial7DoF.
 * The object is immutable after construction and is therefore safe to reuse
 * concurrently for independent arm trajectories.
 */
class RobotModel {
 public:
  std::array<Vec, 7> axes;
  std::array<Mat, 7> local;
  Joints lower, upper;
  Mat tool, align;

  /// Return base<-frame rotation for frame indices in [0, 7].
  Mat r0(const Joints& q, int frame) const {
    Mat r = eye();
    for (int j = 0; j < frame; ++j) r = mul(r, mul(local[j], rot(axes[j], q[j])));
    return r;
  }
  /// Return frame-a<-frame-b rotation.
  Mat between(const Joints& q, int a, int b) const { return mul(transpose(r0(q, a)), r0(q, b)); }
};

std::pair<double, bool> sp1(Vec p1, Vec p2, Vec k) {
  k = unit(k);
  const Vec kxp = cross(k, p1);
  const Vec second = scale(cross(k, kxp), -1.0);
  const double x0 = dot(kxp, p2), x1 = dot(second, p2);
  if (std::hypot(x0, x1) < kEps) return {0.0, true};
  return {std::atan2(x0, x1),
          std::abs(norm(p1) - norm(p2)) > 1e-8 || std::abs(dot(k, p1) - dot(k, p2)) > 1e-8};
}

std::pair<std::vector<double>, bool> sp4(Vec h, Vec p, Vec k, double d) {
  h = unit(h);
  k = unit(k);
  const Vec a11 = cross(k, p);
  const Vec a12 = scale(cross(k, a11), -1.0);
  const double a0 = dot(h, a11), a1 = dot(h, a12);
  const double b = d - dot(h, k) * dot(k, p), n2 = a0 * a0 + a1 * a1;
  if (n2 < kEps) return {{0.0}, std::abs(dot(h, p) - d) > 1e-8};
  const Vec hp = scale(h, b);
  const double x0 = dot(a11, hp), x1 = dot(a12, hp);
  if (n2 > b * b + 1e-12) {
    const double xi = std::sqrt(std::max(0.0, n2 - b * b));
    return {{std::atan2(x0 + xi * a1, x1 - xi * a0), std::atan2(x0 - xi * a1, x1 + xi * a0)},
            false};
  }
  const double theta = std::atan2(x0, x1);
  return {{theta}, std::abs(dot(h, mul(rot(k, theta), p)) - d) > 1e-8};
}

std::pair<std::vector<std::pair<double, double>>, bool> sp2(Vec p1, Vec p2, Vec k1, Vec k2) {
  k1 = unit(k1);
  k2 = unit(k2);
  if (norm(cross(k1, k2)) < kEps) throw std::runtime_error("SP2 axes are parallel");
  const double length_error = std::abs(norm(p1) - norm(p2));
  Vec n1 = unit(p1), n2 = unit(p2);
  auto [t1, ls1] = sp4(k2, n1, k1, dot(k2, n2));
  auto [t2, ls2] = sp4(k1, n2, k2, dot(k1, n1));
  std::vector<std::pair<double, double>> out;
  if (t1.size() > 1 || t2.size() > 1)
    out = {{t1.front(), t2.back()}, {t1.back(), t2.front()}};
  else
    out = {{t1.front(), t2.front()}};
  return {out, length_error > 1e-8 || ls1 || ls2};
}

std::vector<double> equivalent(double theta, double lower, double upper, double reference) {
  const int kmin = static_cast<int>(std::floor((lower - theta) / kTwoPi)) - 1;
  const int kmax = static_cast<int>(std::ceil((upper - theta) / kTwoPi)) + 1;
  std::vector<double> values;
  for (int k = kmin; k <= kmax; ++k) {
    const double value = theta + kTwoPi * k;
    if (value >= lower - kEps && value <= upper + kEps) values.push_back(value);
  }
  std::sort(values.begin(), values.end(), [reference](double a, double b) {
    return std::abs(a - reference) < std::abs(b - reference);
  });
  return values;
}

/** Closed-form, joint-limit-aware solver for one seven-DoF arm.
 *
 * The solver owns a calibrated RobotModel but no mutable trajectory state.
 * Continuity is controlled explicitly by passing the previous q as q0.
 */
class SewMimicSolver {
 public:
  explicit SewMimicSolver(RobotModel robot) : robot_(std::move(robot)) {}

  /** Solve one frame expressed in the robot arm-base coordinate system.
   *
   * Keypoint magnitudes do not need to match robot link lengths: only the
   * normalized shoulder-elbow and elbow-wrist directions are retargeted.
   */
  Joints solve(Joints q, Vec shoulder, Vec elbow, Vec wrist, Mat hand) const {
    if (!finite(shoulder) || !finite(elbow) || !finite(wrist))
      throw std::runtime_error("Shoulder, elbow, and wrist must be finite 3-vectors");
    if (!rotation_matrix(hand))
      throw std::runtime_error("Hand orientation must be a valid rotation matrix");
    if (!std::all_of(q.begin(), q.end(), [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("Initial joint vector must be finite");
    const Vec upper = unit(sub(elbow, shoulder));
    const Vec lower = unit(sub(wrist, elbow));
    std::tie(q[0], q[1]) = alignAxis(3, q, upper);
    std::tie(q[2], q[3]) = alignAxis(5, q, lower);
    const Mat desired = mul(mul(hand, transpose(robot_.align)), transpose(robot_.tool));
    std::tie(q[4], q[5]) = alignAxis(7, q, mul(desired, robot_.axes[6]));
    const Vec u6 = mul(transpose(desired), mul(robot_.r0(q, 6), robot_.axes[5]));
    const Vec h6 = mul(transpose(robot_.local[6]), robot_.axes[5]);
    auto [q7, unused] = sp1(h6, u6, scale(robot_.axes[6], -1.0));
    (void)unused;
    q[6] = boundAngle(q7, 6, q[6]);
    return q;
  }

 private:
  /// Paper Algorithm 2; solve q_(i-2), q_(i-1) for axis i in {3, 5, 7}.
  std::pair<double, double> alignAxis(int i, const Joints& q0, Vec target) const {
    target = unit(target);
    const int ia = i - 3, ib = i - 2, frame = i - 2;
    Joints base = q0;
    base[ia] = 0.0;
    base[ib] = 0.0;
    const Vec vf = mul(transpose(robot_.r0(base, frame)), target);
    const Vec hi = mul(robot_.between(base, frame, i), robot_.axes[i - 1]);
    const Vec hp = mul(robot_.between(base, frame, i - 1), robot_.axes[i - 2]);
    auto [raw, unused] = sp2(vf, hi, scale(robot_.axes[i - 3], -1.0), hp);
    (void)unused;
    bool found = false;
    double best = 0, qa_best = 0, qb_best = 0;
    for (auto [qa, qb] : raw)
      for (double a : equivalent(qa, robot_.lower[ia], robot_.upper[ia], q0[ia]))
        for (double b : equivalent(qb, robot_.lower[ib], robot_.upper[ib], q0[ib])) {
          const double cost = std::abs(q0[ia] - a) + std::abs(q0[ib] - b);
          if (!found || cost < best) {
            found = true;
            best = cost;
            qa_best = a;
            qb_best = b;
          }
        }
    if (!found) throw std::runtime_error("AlignAxis solutions violate joint limits");
    return {qa_best, qb_best};
  }

  /// Select the limit-valid 2*pi equivalent nearest the previous joint angle.
  double boundAngle(double theta, int i, double reference) const {
    auto values = equivalent(theta, robot_.lower[i], robot_.upper[i], reference);
    if (values.empty()) throw std::runtime_error("Wrist solution violates joint limits");
    return values.front();
  }

  RobotModel robot_;
};

/** Allocation-light capsule collision queries in metres.
 *
 * Signed distance is positive for clearance and negative for penetration.
 */
class CapsuleCollisionDetector {
 public:
  struct Capsule {
    Vec start;
    Vec end;
    double radius;
  };

  struct Contact {
    double distance;
    Vec normal;
    double parameter_a;
    double parameter_b;
  };

  static double signedDistance(const Capsule& first, const Capsule& second) {
    return contact(first, second).distance;
  }

  /// Return distance, A-to-B normal, and closest segment interpolation parameters.
  static Contact contact(const Capsule& first, const Capsule& second) {
    const auto closest = closestPoints(first.start, first.end, second.start, second.end);
    const Vec delta = sub(closest.point_b, closest.point_a);
    const double center_distance = norm(delta);
    Vec normal;
    if (center_distance > kEps) {
      normal = scale(delta, 1.0 / center_distance);
    } else {
      normal = cross(sub(first.end, first.start), sub(second.end, second.start));
      if (dot(normal, normal) <= kEps * kEps)
        normal = cross(sub(first.end, first.start), Vec{1.0, 0.0, 0.0});
      if (dot(normal, normal) <= kEps * kEps) normal = {0.0, 1.0, 0.0};
      normal = unit(normal);
    }
    return {center_distance - first.radius - second.radius, normal, closest.parameter_a,
            closest.parameter_b};
  }

  /// Evaluate only configured non-adjacent/self-collision pairs.
  double minimumDistance(const std::vector<Capsule>& capsules,
                         const std::vector<std::pair<int, int>>& pairs) const {
    if (pairs.empty()) throw std::runtime_error("Collision-pair list must not be empty");
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& [first, second] : pairs) {
      if (first < 0 || second < 0 || first >= static_cast<int>(capsules.size()) ||
          second >= static_cast<int>(capsules.size()))
        throw std::runtime_error("Collision-pair index is out of range");
      minimum = std::min(minimum, signedDistance(capsules[first], capsules[second]));
    }
    return minimum;
  }

 private:
  struct ClosestPoints {
    Vec point_a;
    Vec point_b;
    double parameter_a;
    double parameter_b;
  };

  static ClosestPoints closestPoints(Vec a0, Vec a1, Vec b0, Vec b1) {
    const Vec u = sub(a1, a0), v = sub(b1, b0), w = sub(a0, b0);
    const double aa = dot(u, u), bb = dot(u, v), cc = dot(v, v);
    const double dd = dot(u, w), ee = dot(v, w);
    double s = 0.0, t = 0.0;
    auto clamp = [](double value) { return std::clamp(value, 0.0, 1.0); };
    if (aa <= kEps && cc <= kEps) return {a0, b0, 0.0, 0.0};
    if (aa <= kEps) {
      t = clamp(ee / cc);
    } else if (cc <= kEps) {
      s = clamp(-dd / aa);
    } else {
      const double denominator = aa * cc - bb * bb;
      if (denominator > kEps) s = clamp((bb * ee - cc * dd) / denominator);
      t = (bb * s + ee) / cc;
      if (t < 0.0) {
        t = 0.0;
        s = clamp(-dd / aa);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp((bb - dd) / aa);
      }
    }
    return {add(a0, scale(u, s)), add(b0, scale(v, t)), s, t};
  }
};

/** Paper-style XPBD projection over eight bimanual keypoints.
 *
 * Shoulders (indices 0 and 4) and the torso capsule are fixed. All distance
 * values, compliance, and keypoints use SI metres.
 */
class XpbdCollisionProjector {
 public:
  struct Settings {
    double minimum_distance;
    double activation_distance;
    double release_distance;
    double compliance;
    double tolerance;
    int iterations;
  };

  XpbdCollisionProjector(Vec torso_start, Vec torso_end, std::array<double, 4> radii,
                         std::vector<std::pair<int, int>> pairs, Settings settings)
      : torso_start_(torso_start),
        torso_end_(torso_end),
        radii_(radii),
        pairs_(std::move(pairs)),
        settings_(settings) {}

  /// Project until clearance tolerance is reached or the iteration budget expires.
  std::tuple<std::array<Vec, 8>, int, double> project(std::array<Vec, 8> points) const {
    const std::array<std::pair<int, int>, 6> links = {std::pair{0, 1}, {1, 2}, {2, 3},
                                                      {4, 5},          {5, 6}, {6, 7}};
    std::array<double, 6> lengths{};
    for (size_t i = 0; i < links.size(); ++i)
      lengths[i] = norm(sub(points[links[i].second], points[links[i].first]));
    std::vector<double> multipliers(pairs_.size(), 0.0);
    double distance = minimumDistance(points);
    int used_iterations = 0;
    for (used_iterations = 1; used_iterations <= settings_.iterations; ++used_iterations) {
      if (distance >= settings_.minimum_distance - settings_.tolerance) break;
      collisionPass(points, multipliers);
      lengthPass(points, lengths, links);
      distance = minimumDistance(points);
    }
    if (used_iterations > settings_.iterations) used_iterations = settings_.iterations;
    return {points, used_iterations, distance};
  }

 private:
  std::array<CapsuleCollisionDetector::Capsule, 7> capsules(
      const std::array<Vec, 8>& points) const {
    return {{{torso_start_, torso_end_, radii_[0]},
             {points[0], points[1], radii_[1]},
             {points[1], points[2], radii_[2]},
             {points[2], points[3], radii_[3]},
             {points[4], points[5], radii_[1]},
             {points[5], points[6], radii_[2]},
             {points[6], points[7], radii_[3]}}};
  }

  double minimumDistance(const std::array<Vec, 8>& points) const {
    auto current = capsules(points);
    std::vector<CapsuleCollisionDetector::Capsule> vector(current.begin(), current.end());
    return detector_.minimumDistance(vector, pairs_);
  }

  /// Apply one persistent-multiplier inequality-constraint pass.
  void collisionPass(std::array<Vec, 8>& points, std::vector<double>& multipliers) const {
    auto current = capsules(points);
    const std::array<std::pair<int, int>, 6> moving = {std::pair{0, 1}, {1, 2}, {2, 3},
                                                       {4, 5},          {5, 6}, {6, 7}};
    for (size_t pair_index = 0; pair_index < pairs_.size(); ++pair_index) {
      const auto [first, second] = pairs_[pair_index];
      const auto contact = detector_.contact(current[first], current[second]);
      if (contact.distance >= settings_.release_distance ||
          (contact.distance >= settings_.activation_distance && multipliers[pair_index] == 0.0)) {
        multipliers[pair_index] = 0.0;
        continue;
      }
      const double constraint = contact.distance - settings_.minimum_distance;
      if (constraint >= 0.0) continue;
      std::array<Vec, 8> gradients{};
      std::array<bool, 8> active{};
      for (const auto& [capsule_index, sign, parameter] :
           {std::tuple<int, double, double>{first, -1.0, contact.parameter_a},
            std::tuple<int, double, double>{second, 1.0, contact.parameter_b}}) {
        if (capsule_index == 0) continue;
        const auto [start, end] = moving[capsule_index - 1];
        gradients[start] = add(gradients[start], scale(contact.normal, sign * (1.0 - parameter)));
        gradients[end] = add(gradients[end], scale(contact.normal, sign * parameter));
        active[start] = active[end] = true;
      }
      double denominator = settings_.compliance;
      for (int i = 0; i < 8; ++i)
        if (active[i] && i != 0 && i != 4) denominator += dot(gradients[i], gradients[i]);
      const double old = multipliers[pair_index];
      const double increment =
          -(constraint + settings_.compliance * old) / std::max(denominator, kEps);
      const double updated = std::max(0.0, old + increment);
      multipliers[pair_index] = updated;
      for (int i = 0; i < 8; ++i)
        if (active[i] && i != 0 && i != 4)
          points[i] = add(points[i], scale(gradients[i], updated - old));
      current = capsules(points);
    }
  }

  /// Restore the six shoulder-elbow-wrist-tool segment lengths after collision projection.
  void lengthPass(std::array<Vec, 8>& points, const std::array<double, 6>& lengths,
                  const std::array<std::pair<int, int>, 6>& links) const {
    for (size_t i = 0; i < links.size(); ++i) {
      const auto [start, end] = links[i];
      const Vec delta = sub(points[end], points[start]);
      const double length = norm(delta);
      if (length <= kEps) continue;
      const double weight_start = (start == 0 || start == 4) ? 0.0 : 1.0;
      const double correction =
          -(length - lengths[i]) / (settings_.compliance + weight_start + 1.0);
      const Vec gradient = scale(delta, 1.0 / length);
      points[start] = sub(points[start], scale(gradient, weight_start * correction));
      points[end] = add(points[end], scale(gradient, correction));
    }
  }

  Vec torso_start_, torso_end_;
  std::array<double, 4> radii_;
  std::vector<std::pair<int, int>> pairs_;
  Settings settings_;
  CapsuleCollisionDetector detector_;
};

template <size_t N>
std::array<double, N> vector_array(const DoubleArray& input, const char* name) {
  auto a = input.request();
  if (a.size != N) throw py::value_error(std::string(name) + " has an invalid size");
  const auto* p = static_cast<const double*>(a.ptr);
  std::array<double, N> out{};
  std::copy(p, p + N, out.begin());
  return out;
}

RobotModel make_robot(const DoubleArray& axes, const DoubleArray& local, const DoubleArray& lower,
                      const DoubleArray& upper, const DoubleArray& tool, const DoubleArray& align) {
  RobotModel robot;
  auto av = vector_array<21>(axes, "axes_local");
  auto rv = vector_array<63>(local, "R_local");
  for (int i = 0; i < 7; ++i) {
    std::copy(av.begin() + 3 * i, av.begin() + 3 * i + 3, robot.axes[i].begin());
    std::copy(rv.begin() + 9 * i, rv.begin() + 9 * i + 9, robot.local[i].begin());
    robot.axes[i] = unit(robot.axes[i]);
    if (!rotation_matrix(robot.local[i]))
      throw py::value_error("Every R_local entry must be a valid rotation matrix");
  }
  robot.lower = vector_array<7>(lower, "q_min");
  robot.upper = vector_array<7>(upper, "q_max");
  robot.tool = vector_array<9>(tool, "R_7T_local");
  robot.align = vector_array<9>(align, "R_align");
  if (!rotation_matrix(robot.tool) || !rotation_matrix(robot.align))
    throw py::value_error("Tool and alignment transforms must be valid rotation matrices");
  for (int i = 0; i < 7; ++i)
    if (!std::isfinite(robot.lower[i]) || !std::isfinite(robot.upper[i]) ||
        robot.lower[i] > robot.upper[i])
      throw py::value_error("Joint limits must be finite and ordered");
  return robot;
}

py::array_t<double> solve_with(const SewMimicSolver& solver, const DoubleArray& q0,
                               const DoubleArray& shoulder, const DoubleArray& elbow,
                               const DoubleArray& wrist, const DoubleArray& hand) {
  const Joints current = vector_array<7>(q0, "q0");
  const Vec shoulder_value = vector_array<3>(shoulder, "shoulder");
  const Vec elbow_value = vector_array<3>(elbow, "elbow");
  const Vec wrist_value = vector_array<3>(wrist, "wrist");
  const Mat hand_value = vector_array<9>(hand, "hand_orientation");
  Joints result;
  {
    py::gil_scoped_release release;
    result = solver.solve(current, shoulder_value, elbow_value, wrist_value, hand_value);
  }
  py::array_t<double> output(7);
  std::copy(result.begin(), result.end(), output.mutable_data());
  return output;
}

py::array_t<double> solve_batch_with(const SewMimicSolver& solver, const DoubleArray& q0,
                                     const DoubleArray& shoulders, const DoubleArray& elbows,
                                     const DoubleArray& wrists, const DoubleArray& hands) {
  const auto s = shoulders.request(), e = elbows.request(), w = wrists.request(),
             h = hands.request();
  if (s.ndim != 2 || s.shape[1] != 3 || e.ndim != 2 || w.ndim != 2 || e.shape[1] != 3 ||
      w.shape[1] != 3 || e.shape[0] != s.shape[0] || w.shape[0] != s.shape[0] || h.ndim != 3 ||
      h.shape[0] != s.shape[0] || h.shape[1] != 3 || h.shape[2] != 3)
    throw py::value_error("Batch inputs must have shapes (N,3), (N,3,3)");
  const auto* sp = static_cast<const double*>(s.ptr);
  const auto* ep = static_cast<const double*>(e.ptr);
  const auto* wp = static_cast<const double*>(w.ptr);
  const auto* hp = static_cast<const double*>(h.ptr);
  py::array_t<double> output({s.shape[0], py::ssize_t{7}});
  auto* out = output.mutable_data();
  Joints current = vector_array<7>(q0, "q0");
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < s.shape[0]; ++i) {
      Vec shoulder{}, elbow{}, wrist{};
      Mat hand{};
      std::copy(sp + 3 * i, sp + 3 * i + 3, shoulder.begin());
      std::copy(ep + 3 * i, ep + 3 * i + 3, elbow.begin());
      std::copy(wp + 3 * i, wp + 3 * i + 3, wrist.begin());
      std::copy(hp + 9 * i, hp + 9 * i + 9, hand.begin());
      current = solver.solve(current, shoulder, elbow, wrist, hand);
      std::copy(current.begin(), current.end(), out + 7 * i);
    }
  }
  return output;
}

py::array_t<double> solve(const DoubleArray& axes, const DoubleArray& local,
                          const DoubleArray& lower, const DoubleArray& upper,
                          const DoubleArray& tool, const DoubleArray& align, const DoubleArray& q0,
                          const DoubleArray& shoulder, const DoubleArray& elbow,
                          const DoubleArray& wrist, const DoubleArray& hand) {
  return solve_with(SewMimicSolver(make_robot(axes, local, lower, upper, tool, align)), q0,
                    shoulder, elbow, wrist, hand);
}

double minimum_capsule_distance(const DoubleArray& starts, const DoubleArray& ends,
                                const DoubleArray& radii, const IntArray& pairs) {
  const auto start_info = starts.request(), end_info = ends.request();
  const auto radius_info = radii.request(), pair_info = pairs.request();
  if (start_info.ndim != 2 || start_info.shape[1] != 3 || end_info.ndim != 2 ||
      end_info.shape[1] != 3 || end_info.shape[0] != start_info.shape[0] || radius_info.ndim != 1 ||
      radius_info.shape[0] != start_info.shape[0] || pair_info.ndim != 2 || pair_info.shape[1] != 2)
    throw py::value_error("Invalid capsule array shapes");
  const auto* start_data = static_cast<const double*>(start_info.ptr);
  const auto* end_data = static_cast<const double*>(end_info.ptr);
  const auto* radius_data = static_cast<const double*>(radius_info.ptr);
  const auto* pair_data = static_cast<const int*>(pair_info.ptr);
  std::vector<CapsuleCollisionDetector::Capsule> capsules(start_info.shape[0]);
  for (py::ssize_t i = 0; i < start_info.shape[0]; ++i) {
    std::copy(start_data + 3 * i, start_data + 3 * i + 3, capsules[i].start.begin());
    std::copy(end_data + 3 * i, end_data + 3 * i + 3, capsules[i].end.begin());
    capsules[i].radius = radius_data[i];
    if (!std::isfinite(capsules[i].radius) || capsules[i].radius <= 0.0)
      throw py::value_error("Capsule radii must be finite and positive");
  }
  std::vector<std::pair<int, int>> collision_pairs;
  collision_pairs.reserve(pair_info.shape[0]);
  for (py::ssize_t i = 0; i < pair_info.shape[0]; ++i)
    collision_pairs.emplace_back(pair_data[2 * i], pair_data[2 * i + 1]);
  double distance;
  {
    py::gil_scoped_release release;
    distance = CapsuleCollisionDetector().minimumDistance(capsules, collision_pairs);
  }
  return distance;
}

py::tuple project_xpbd(const DoubleArray& points, const DoubleArray& torso_start,
                       const DoubleArray& torso_end, const DoubleArray& radii,
                       const IntArray& pairs, double minimum_distance, double activation_distance,
                       double release_distance, double compliance, double tolerance,
                       int iterations) {
  const auto point_info = points.request(), pair_info = pairs.request();
  if (point_info.ndim != 2 || point_info.shape[0] != 8 || point_info.shape[1] != 3 ||
      pair_info.ndim != 2 || pair_info.shape[1] != 2)
    throw py::value_error("XPBD points must be (8,3) and pairs must be (N,2)");
  auto flat_points = vector_array<24>(points, "points");
  std::array<Vec, 8> native_points{};
  for (int i = 0; i < 8; ++i)
    std::copy(flat_points.begin() + 3 * i, flat_points.begin() + 3 * i + 3,
              native_points[i].begin());
  auto radius_values = vector_array<4>(radii, "radii");
  const auto* pair_data = static_cast<const int*>(pair_info.ptr);
  std::vector<std::pair<int, int>> collision_pairs;
  for (py::ssize_t i = 0; i < pair_info.shape[0]; ++i)
    collision_pairs.emplace_back(pair_data[2 * i], pair_data[2 * i + 1]);
  XpbdCollisionProjector::Settings settings{minimum_distance, activation_distance, release_distance,
                                            compliance,       tolerance,           iterations};
  XpbdCollisionProjector projector(vector_array<3>(torso_start, "torso_start"),
                                   vector_array<3>(torso_end, "torso_end"), radius_values,
                                   collision_pairs, settings);
  std::array<Vec, 8> projected;
  int used_iterations;
  double distance;
  {
    py::gil_scoped_release release;
    std::tie(projected, used_iterations, distance) = projector.project(native_points);
  }
  py::array_t<double> output({py::ssize_t{8}, py::ssize_t{3}});
  for (int i = 0; i < 8; ++i)
    std::copy(projected[i].begin(), projected[i].end(), output.mutable_data() + 3 * i);
  return py::make_tuple(output, used_iterations, distance);
}

PYBIND11_MODULE(_sew_mimic_cpp, m) {
  m.doc() = "Native C++17 SEW-Mimic solver";
  m.attr("implementation") = "native";
  py::class_<SewMimicSolver>(m, "SewMimicSolver")
      .def(
          py::init([](const DoubleArray& axes, const DoubleArray& local, const DoubleArray& lower,
                      const DoubleArray& upper, const DoubleArray& tool, const DoubleArray& align) {
            return SewMimicSolver(make_robot(axes, local, lower, upper, tool, align));
          }),
          py::arg("axes_local"), py::arg("rotations_local"), py::arg("q_min"), py::arg("q_max"),
          py::arg("tool_rotation"), py::arg("alignment"),
          "Construct and validate one reusable native robot solver.")
      .def("solve", &solve_with, py::arg("q0"), py::arg("shoulder"), py::arg("elbow"),
           py::arg("wrist"), py::arg("hand_orientation"),
           "Solve one arm frame; keypoints are expressed in the arm-base frame.")
      .def("solve_batch", &solve_batch_with, py::arg("q0"), py::arg("shoulders"), py::arg("elbows"),
           py::arg("wrists"), py::arg("hand_orientations"),
           "Solve an ordered trajectory, carrying each result into the next frame.");
  m.def("solve", &solve, py::arg("axes_local"), py::arg("rotations_local"), py::arg("q_min"),
        py::arg("q_max"), py::arg("tool_rotation"), py::arg("alignment"), py::arg("q0"),
        py::arg("shoulder"), py::arg("elbow"), py::arg("wrist"), py::arg("hand_orientation"),
        "Construct a temporary model and solve one arm frame.");
  m.def("minimum_capsule_distance", &minimum_capsule_distance, py::arg("starts"), py::arg("ends"),
        py::arg("radii"), py::arg("pairs"),
        "Return minimum signed capsule distance in metres for configured pairs.");
  m.def("project_xpbd", &project_xpbd, py::arg("points"), py::arg("torso_start"),
        py::arg("torso_end"), py::arg("radii"), py::arg("pairs"), py::arg("minimum_distance"),
        py::arg("activation_distance"), py::arg("release_distance"), py::arg("compliance"),
        py::arg("tolerance"), py::arg("iterations"),
        "Project eight bimanual keypoints and return (points, iterations, distance).");
}
