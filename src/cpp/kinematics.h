#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

#include "math_utils.h"

// This executor consumes the same folded tree as the Python FK implementation.
// URDF parsing, name resolution and fixed-chain folding remain in Python.
class ForwardKinematics {
  using DoubleArray =
      pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;
  using IntArray = pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast>;

  struct Transform {
    Mat rotation;
    Vec translation;
  };
  struct Operation {
    int parent, motion, index;
    Transform origin;
    Vec axis;
  };

  static Transform compose(const Transform& a, const Transform& b) {
    return {mul(a.rotation, b.rotation), add(a.translation, mul(a.rotation, b.translation))};
  }

  static Transform read_transform(const double* values) {
    Transform result;
    for (int row = 0; row < 3; ++row) {
      result.translation[row] = values[4 * row + 3];
      for (int col = 0; col < 3; ++col) result.rotation[3 * row + col] = values[4 * row + col];
    }
    if (!rotation_matrix(result.rotation) || !finite(result.translation) || values[12] != 0.0 ||
        values[13] != 0.0 || values[14] != 0.0 || values[15] != 1.0)
      throw pybind11::value_error("FK transforms must be finite rigid transforms");
    return result;
  }

 public:
  ForwardKinematics(const IntArray& operations, const DoubleArray& origins, const DoubleArray& axes,
                    const IntArray& targets, const DoubleArray& offsets, int joint_count)
      : joint_count_(joint_count) {
    const auto op = operations.request(), org = origins.request(), ax = axes.request();
    const auto tg = targets.request(), off = offsets.request();
    if (joint_count < 0 || op.ndim != 2 || op.shape[1] != 3 || org.ndim != 3 ||
        org.shape[0] != op.shape[0] || org.shape[1] != 4 || org.shape[2] != 4 || ax.ndim != 2 ||
        ax.shape[0] != op.shape[0] || ax.shape[1] != 3 || tg.ndim != 1 || off.ndim != 3 ||
        off.shape[0] != tg.shape[0] || off.shape[1] != 4 || off.shape[2] != 4)
      throw pybind11::value_error("Invalid FK plan shapes");
    const auto* op_data = static_cast<const int*>(op.ptr);
    const auto* origin_data = static_cast<const double*>(org.ptr);
    const auto* axis_data = static_cast<const double*>(ax.ptr);
    for (pybind11::ssize_t i = 0; i < op.shape[0]; ++i) {
      const int parent = op_data[3 * i], motion = op_data[3 * i + 1], index = op_data[3 * i + 2];
      if (parent < 0 || parent > i || (motion != 1 && motion != 2) || index < 0 ||
          index >= joint_count)
        throw pybind11::value_error("Invalid FK operation indices or motion type");
      Vec axis{axis_data[3 * i], axis_data[3 * i + 1], axis_data[3 * i + 2]};
      if (!finite(axis) || norm(axis) < kEps)
        throw pybind11::value_error("FK axes must be finite and nonzero");
      operations_.push_back(
          {parent, motion, index, read_transform(origin_data + 16 * i), unit(axis)});
    }
    const auto* target_data = static_cast<const int*>(tg.ptr);
    const auto* offset_data = static_cast<const double*>(off.ptr);
    for (pybind11::ssize_t i = 0; i < tg.shape[0]; ++i) {
      if (target_data[i] < 0 || target_data[i] > op.shape[0])
        throw pybind11::value_error("FK target index is out of range");
      targets_.push_back(target_data[i]);
      offsets_.push_back(read_transform(offset_data + 16 * i));
    }
  }

  pybind11::array_t<double> evaluate(const DoubleArray& q) const {
    const auto info = q.request();
    if (info.ndim != 1 || info.shape[0] != joint_count_)
      throw pybind11::value_error("q has an invalid shape for the FK plan");
    const auto* data = static_cast<const double*>(info.ptr);
    // Own the command before releasing the GIL; model and scratch are separate.
    const std::vector<double> values(data, data + joint_count_);
    for (double value : values)
      if (!std::isfinite(value)) throw pybind11::value_error("q must contain only finite values");
    pybind11::array_t<double> result({static_cast<pybind11::ssize_t>(targets_.size()),
                                      pybind11::ssize_t{4}, pybind11::ssize_t{4}});
    auto* output = result.mutable_data();
    {
      pybind11::gil_scoped_release release;
      std::vector<Transform> world(operations_.size() + 1);
      world[0] = {eye(), {0, 0, 0}};
      for (size_t i = 0; i < operations_.size(); ++i) {
        const auto& op = operations_[i];
        Transform local = op.origin;
        if (op.motion == 1)
          local.rotation = mul(local.rotation, rot(op.axis, values[op.index]));
        else
          local.translation =
              add(local.translation, mul(local.rotation, scale(op.axis, values[op.index])));
        world[i + 1] = compose(world[op.parent], local);
      }
      for (size_t i = 0; i < targets_.size(); ++i) {
        const auto transform = compose(world[targets_[i]], offsets_[i]);
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col)
            output[16 * i + 4 * row + col] = transform.rotation[3 * row + col];
          output[16 * i + 4 * row + 3] = transform.translation[row];
        }
        output[16 * i + 12] = output[16 * i + 13] = output[16 * i + 14] = 0.0;
        output[16 * i + 15] = 1.0;
      }
    }
    return result;
  }

 private:
  int joint_count_;
  std::vector<Operation> operations_;
  std::vector<int> targets_;
  std::vector<Transform> offsets_;
};

inline void bind_kinematics(pybind11::module_& module) {
  using DoubleArray =
      pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;
  using IntArray = pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast>;
  pybind11::class_<ForwardKinematics>(module, "ForwardKinematics")
      .def(pybind11::init<const IntArray&, const DoubleArray&, const DoubleArray&, const IntArray&,
                          const DoubleArray&, int>())
      .def("evaluate", &ForwardKinematics::evaluate, pybind11::arg("q"));
}
