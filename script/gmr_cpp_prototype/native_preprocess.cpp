#include <cmath>
#include <cstddef>

namespace {

struct Quaternion {
  double w;
  double x;
  double y;
  double z;
};

Quaternion multiply(const Quaternion& a, const Quaternion& b) {
  return {
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

Quaternion normalized(const Quaternion& q) {
  const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (norm <= 1e-15) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

void rotate(const Quaternion& q, const double* vector, double* output) {
  const double tx = 2.0 * (q.y * vector[2] - q.z * vector[1]);
  const double ty = 2.0 * (q.z * vector[0] - q.x * vector[2]);
  const double tz = 2.0 * (q.x * vector[1] - q.y * vector[0]);
  output[0] = vector[0] + q.w * tx + (q.y * tz - q.z * ty);
  output[1] = vector[1] + q.w * ty + (q.z * tx - q.x * tz);
  output[2] = vector[2] + q.w * tz + (q.x * ty - q.y * tx);
}

}  // namespace

extern "C" int gmr_preprocess(
    const double* positions,
    const double* quaternions_wxyz,
    const double* scales,
    const double* position_offsets,
    const double* rotation_offsets_wxyz,
    std::size_t count,
    std::size_t root_index,
    double ground_offset,
    double* output_positions,
    double* output_quaternions_wxyz) {
  if (positions == nullptr || quaternions_wxyz == nullptr || scales == nullptr ||
      position_offsets == nullptr || rotation_offsets_wxyz == nullptr ||
      output_positions == nullptr || output_quaternions_wxyz == nullptr ||
      count == 0 || root_index >= count) {
    return 1;
  }

  const double* root_position = positions + 3 * root_index;
  const double scaled_root[3] = {
      scales[root_index] * root_position[0],
      scales[root_index] * root_position[1],
      scales[root_index] * root_position[2],
  };

  for (std::size_t index = 0; index < count; ++index) {
    const double* input_position = positions + 3 * index;
    const double* input_quaternion = quaternions_wxyz + 4 * index;
    const double* rotation_offset = rotation_offsets_wxyz + 4 * index;

    double scaled_position[3];
    if (index == root_index) {
      scaled_position[0] = scaled_root[0];
      scaled_position[1] = scaled_root[1];
      scaled_position[2] = scaled_root[2];
    } else {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        scaled_position[axis] =
            (input_position[axis] - root_position[axis]) * scales[index] + scaled_root[axis];
      }
    }

    const Quaternion input = {
        input_quaternion[0], input_quaternion[1], input_quaternion[2], input_quaternion[3]};
    const Quaternion offset = {
        rotation_offset[0], rotation_offset[1], rotation_offset[2], rotation_offset[3]};
    const Quaternion updated = normalized(multiply(input, offset));

    double rotated_offset[3];
    rotate(updated, position_offsets + 3 * index, rotated_offset);
    double* output_position = output_positions + 3 * index;
    output_position[0] = scaled_position[0] + rotated_offset[0];
    output_position[1] = scaled_position[1] + rotated_offset[1];
    output_position[2] = scaled_position[2] + rotated_offset[2] - ground_offset;

    double* output_quaternion = output_quaternions_wxyz + 4 * index;
    output_quaternion[0] = updated.w;
    output_quaternion[1] = updated.x;
    output_quaternion[2] = updated.y;
    output_quaternion[3] = updated.z;
  }
  return 0;
}
