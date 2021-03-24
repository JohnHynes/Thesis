#ifndef GPU_RAY_TRACER_HITTABLE_HPP_
#define GPU_RAY_TRACER_HITTABLE_HPP_

#include <algorithm>
#include <glm/geometric.hpp>
#include <iostream>
#include <iterator>
#include <limits>

#include "Preprocessor.hpp"
#include "Random.hpp"
#include "Ray.hpp"
#include "constants.hpp"
#include "types.hpp"

struct hittable;

struct hit_record {
  point3 point;
  vec3 normal;
  int mat_idx;
  num t;
  bool front_face;

  __host__ __device__ inline void set_face_normal(const ray &r,
                                                  const vec3 &outward_normal) {
    front_face = glm::dot(r.dir, outward_normal) < CONST(0.0);
    if (front_face) {
      normal = outward_normal;
    } else {
      normal = -outward_normal;
    }
  }
};

enum class hittable_id {
  Sphere,
  Plane,
  Rectangle,
  Triangle,
  BoundingBox,
  BoundingArrayNode,
  Unknown
};

struct bounding_box;

struct IHittable {
  __host__ __device__ virtual bool hit(const ray &r, num tmin, num tmax,
                                       hit_record &hitrec) const = 0;

  __host__ virtual inline bounding_box get_bounding_box() const = 0;
};

struct bounding_box : public IHittable {
public:
  point3 minimum, maximum;

private:
  __host__ bounding_box() : minimum(), maximum() {}
  friend class bounding_tree_node_node;

public:
  __host__ __device__ bounding_box(point3 min, point3 max)
      : minimum(min), maximum(max) {}

  __host__ __device__ inline bool hit(const ray &r, num t_min, num t_max,
                                      hit_record &hit_rec) const override {
    for (int i = 0; i < 3; i++) {
      auto invD = CONST(1) / r.dir[i];
      auto t0 = (minimum[i] - r.orig[i]) * invD;
      auto t1 = (maximum[i] - r.orig[i]) * invD;
      if (invD < CONST(0)) {
        auto temp = t0;
        t0 = t1;
        t1 = temp;
      }
      t_min = t0 > t_min ? t0 : t_min;
      t_max = t1 < t_max ? t1 : t_max;
      if (t_max <= t_min) {
        return false;
      }
    }
    return true;
  }

  __host__ inline bounding_box get_bounding_box() const override {
    return *this;
  }
};

__host__ inline bool axis_compare(const bounding_box a, const bounding_box b,
                                  int axis) {
  return a.minimum[axis] < b.minimum[axis];
}

__host__ inline bool x_compare(const bounding_box a, const bounding_box b) {
  return axis_compare(a, b, 0);
}

__host__ inline bool y_compare(const bounding_box a, const bounding_box b) {
  return axis_compare(a, b, 1);
}

__host__ inline bool z_compare(const bounding_box a, const bounding_box b) {
  return axis_compare(a, b, 2);
}

struct bounding_tree_node {
  virtual __host__ bounding_box get_bounding_box() const = 0;
};

struct bounding_tree_node_object : public bounding_tree_node {
  hittable *obj;

  virtual __host__ bounding_box get_bounding_box() const override;
};

struct bounding_tree_node_node : public bounding_tree_node {
public:
  bounding_tree_node *left, *right;
  bounding_box box;

public:
  __host__ bounding_box surrounding_box(bounding_box b1, bounding_box b2) {
    point3 min, max;
    for (int i = 0; i < 3; ++i) {
      min[i] = std::min(b1.minimum[i], b2.minimum[i]);
      max[i] = std::max(b1.maximum[i], b2.maximum[i]);
    }
    return bounding_box(min, max);
  }

  __host__ bounding_tree_node_node(hittable **hittables, int hittables_size,
                                   RandomState *state, int start, int end);

  virtual __host__ inline bounding_box get_bounding_box() const override {
    return box;
  }
};

struct bounding_array_node : public IHittable {
public:
  int left, right;
  bounding_box box;

public:
  __host__ __device__ bounding_array_node(int l, int r, bounding_box b)
      : left(l), right(r), box(b) {}

  __host__ __device__ inline bool hit(const ray &r, num t_min, num t_max,
                                      hit_record &hit_rec) const override {
    return box.hit(r, t_min, t_max, hit_rec);
  }

  __host__ inline bounding_box get_bounding_box() const override { return box; }
};

struct sphere : public IHittable {
public:
  vec3 center;
  num radius;
  int mat_idx;

public:
  __host__ __device__ inline sphere(const vec3 &c, num r, int m)
      : center(c), radius(r), mat_idx(m) {}

  // Member Functions
  __host__ __device__ inline bool hit(const ray &r, num tmin, num tmax,
                                      hit_record &hitrec) const override {
    vec3 oc = r.origin() - center;
    num a = glm::dot(r.dir, r.dir);
    num half_b = glm::dot(oc, r.dir);
    num c = glm::dot(oc, oc) - radius * radius;
    num discriminant = half_b * half_b - a * c;

    if (discriminant <= 0) {
      return false;
    }

    num root = sqrtf(discriminant);

    num t = (-half_b - root) / a;
    if (tmin < t && t < tmax) {
      hitrec.t = t;
      hitrec.point = r.at(hitrec.t);
      glm::vec3 outward_normal = (hitrec.point - center) / radius;
      hitrec.set_face_normal(r, outward_normal);
      hitrec.mat_idx = mat_idx;
      return true;
    }

    t = (-half_b + root) / a;
    if (tmin < t && t < tmax) {
      hitrec.t = t;
      hitrec.point = r.at(hitrec.t);
      glm::vec3 outward_normal = (hitrec.point - center) / radius;
      hitrec.set_face_normal(r, outward_normal);
      hitrec.mat_idx = mat_idx;
      return true;
    }
    return false;
  }

  __host__ inline bounding_box get_bounding_box() const override {
    return bounding_box(center - vec3(radius, radius, radius),
                        center + vec3(radius, radius, radius));
  }
};

struct plane : public IHittable {
public:
  point3 p;
  vec3 surface_normal;
  int mat_idx;

public:
  __host__ __device__ plane(point3 p, vec3 sn, int m)
      : p(p), surface_normal(-glm::normalize(sn)), mat_idx(m) {}

  __host__ __device__ inline bool hit(const ray &r, num t_min, num t_max,
                                      hit_record &rec) const override {
    num numerator = glm::dot(p - r.orig, surface_normal);
    num t = numerator / glm::dot(r.dir, surface_normal);
    if (t < t_min || t_max < t) {
      return false;
    }
    rec.t = t;
    rec.point = r.at(t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal(r, surface_normal);
    return true;
  }

  __host__ inline bounding_box get_bounding_box() const override {
    point3 min, max;
    bool zerox = surface_normal.x == 0;
    bool zeroy = surface_normal.y == 0;
    bool zeroz = surface_normal.z == 0;
    // Check for bounding box infinitely-sized in 3D
    bool atLeastTwoZeros = zerox ? (zeroy || zeroz) : (zeroy && zeroz);
    // If there are at least two zeros, the bounding box is not infinite in 3D
    if (atLeastTwoZeros) {
      if (!zerox) // y-z plane
      {
        min = point3(p.x, -infinity, -infinity);
        max = point3(p.x, infinity, infinity);
      } else if (!zeroy) // x-z plane
      {
        min = point3(-infinity, p.y, -infinity);
        max = point3(infinity, p.y, infinity);
      } else if (!zeroz) // x-y plane
      {
        min = point3(-infinity, -infinity, p.z);
        max = point3(infinity, infinity, p.z);
      } else // Bounding box size of 0x0x0
      {
        min = p;
        max = p;
      }
    }
    // Else, the bounding box is infinite in 3D
    else {
      min = point3(-infinity, -infinity, -infinity);
      max = point3(infinity, infinity, infinity);
    }

    // If not infinitely-sized in 3D, bounding box if infinitely-sized in 2D
    return bounding_box(min, max);
  }
};

struct rectangle : public IHittable {
public:
  int mat_idx;
  point3 p1, p2, p3;
  vec3 surface_normal;

public:
  __host__ __device__ rectangle(point3 _p1, point3 _p2, point3 _p3, int m)
      : mat_idx(m), p1(_p1), p2(_p2), p3(_p3),
        surface_normal(glm::normalize(glm::cross(p1 - p2, p3 - p2))) {}

  __host__ __device__ inline bool hit(const ray &r, num t_min, num t_max,
                                      hit_record &rec) const override {
    // TODO : Implement hit function.
    return false;
  }

  __host__ inline bounding_box get_bounding_box() const override {
    return bounding_box(point3(0.0f, 0.0f, 0.0f), point3(0.0f, 0.0f, 0.0f));
  }
};

struct triangle : public IHittable {
public:
  int mat_idx;
  point3 p1, p2, p3;
  vec3 surface_normal;

public:
  __host__ __device__ triangle(point3 _p1, point3 _p2, point3 _p3, int m)
      : mat_idx(m), p1(_p1), p2(_p2), p3(_p3),
        surface_normal(glm::normalize(glm::cross(p1 - p2, p3 - p2))) {}

  __host__ __device__ virtual bool hit(const ray &r, num t_min, num t_max,
                                       hit_record &rec) const override {
    // TODO : Implement hit function.
    // Use the Moller-Trumbore ray-triangle algorithm to compute t
    num epsilon = CONST(0.0000001);
    vec3 edge1 = p2 - p1;
    vec3 edge2 = p3 - p1;
    vec3 h = glm::cross(r.dir, edge2);
    num a = glm::dot(h, edge1);
    if (-epsilon < a && a < epsilon) // Ray is approximately parallel
    {
      return false;
    }
    num f = CONST(1.0) / a;
    vec3 s = r.orig - p1;
    num u = f * glm::dot(h, s);
    if (u < 0 || 1 < u) {
      return false;
    }
    vec3 q = glm::cross(s, edge1);
    num v = f * glm::dot(r.dir, q);
    if (v < CONST(0) || CONST(1) < u + v) {
      return false;
    }
    num t = f * glm::dot(edge2, q);
    if (t < t_min || t_max < t) {
      return false;
    }
    // 2. Assign proper values to hitrec
    rec.t = t;
    rec.point = r.at(t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal(r, surface_normal);
    return true;
  }

  __host__ inline bounding_box get_bounding_box() const override {
    point3 min, max;
    for (int i = 0; i < 3; ++i) {
      min[i] = std::min({p1[i], p2[i], p3[i]});
      max[i] = std::max({p1[i], p2[i], p3[i]});
    }
    return bounding_box(min, max);
  }
};

struct null_hittable : public IHittable {

  __host__ __device__ null_hittable() {}

  __host__ __device__ inline bool hit(const ray &r, num t_min, num t_max,
                                      hit_record &rec) const override {
    return false;
  }

  __host__ inline bounding_box get_bounding_box() const override {
    // Returning point-sized bounding box.
    return bounding_box(point3(0.0f, 0.0f, 0.0f), point3(0.0f, 0.0f, 0.0f));
  }
};

union hittable_data {
  sphere s;
  plane p;
  rectangle r;
  triangle t;
  bounding_box bb;
  bounding_array_node ban;
  null_hittable n;

  __host__ __device__ inline hittable_data(sphere const &s) : s(s) {}
  __host__ __device__ inline hittable_data(plane const &p) : p(p) {}
  __host__ __device__ inline hittable_data(rectangle const &r) : r(r) {}
  __host__ __device__ inline hittable_data(triangle const &t) : t(t) {}
  __host__ __device__ inline hittable_data(bounding_box const &bb) : bb(bb) {}
  __host__ __device__ inline hittable_data(bounding_array_node const &ban)
      : ban(ban) {}

  __host__ __device__ inline hittable_data(sphere &&s) : s(std::move(s)) {}
  __host__ __device__ inline hittable_data(plane &&p) : p(std::move(p)) {}
  __host__ __device__ inline hittable_data(rectangle &&r) : r(std::move(r)) {}
  __host__ __device__ inline hittable_data(triangle &&t) : t(std::move(t)) {}
  __host__ __device__ inline hittable_data(bounding_box &&bb)
      : bb(std::move(bb)) {}
  __host__ __device__ inline hittable_data(bounding_array_node &&ban)
      : ban(std::move(ban)) {}

  __host__ __device__ inline hittable_data() : n() {}

  template <typename... Args>
  __host__ __device__ inline bool hit(hittable_id id, Args &&...args) const {
    switch (id) {
    case hittable_id::Sphere:
      return s.hit(std::forward<Args>(args)...);
    case hittable_id::Plane:
      return p.hit(std::forward<Args>(args)...);
    case hittable_id::Rectangle:
      return r.hit(std::forward<Args>(args)...);
    case hittable_id::Triangle:
      return t.hit(std::forward<Args>(args)...);
    case hittable_id::BoundingBox:
      return bb.hit(std::forward<Args>(args)...);
    case hittable_id::BoundingArrayNode:
      return ban.hit(std::forward<Args>(args)...);
    default:
      return n.hit(std::forward<Args>(args)...);
    }
  }

  __host__ __device__ inline bounding_box
  get_bounding_box(hittable_id id) const {
    switch (id) {
    case hittable_id::Sphere:
      return s.get_bounding_box();
    case hittable_id::Plane:
      return p.get_bounding_box();
    case hittable_id::Rectangle:
      return r.get_bounding_box();
    case hittable_id::Triangle:
      return t.get_bounding_box();
    case hittable_id::BoundingBox:
      return bb.get_bounding_box();
    case hittable_id::BoundingArrayNode:
      return ban.get_bounding_box();
    default:
      return n.get_bounding_box();
    }
  }

  __host__ __device__ bounding_array_node const &
  as_bounding_array_node() const {
    return ban;
  }
};

struct hittable {
  __host__ __device__ inline hittable(sphere const &s)
      : id(hittable_id::Sphere), data(s) {}
  __host__ __device__ inline hittable(plane const &p)
      : id(hittable_id::Plane), data(p) {}
  __host__ __device__ inline hittable(rectangle const &r)
      : id(hittable_id::Rectangle), data(r) {}
  __host__ __device__ inline hittable(triangle const &t)
      : id(hittable_id::Triangle), data(t) {}
  __host__ __device__ inline hittable(bounding_box const &bb)
      : id(hittable_id::BoundingBox), data(bb) {}
  __host__ __device__ inline hittable(bounding_array_node const &ban)
      : id(hittable_id::BoundingArrayNode), data(ban) {}

  __host__ __device__ inline hittable(sphere &&s)
      : id(hittable_id::Sphere), data(std::move(s)) {}
  __host__ __device__ inline hittable(plane &&p)
      : id(hittable_id::Plane), data(std::move(p)) {}
  __host__ __device__ inline hittable(rectangle &&r)
      : id(hittable_id::Rectangle), data(std::move(r)) {}
  __host__ __device__ inline hittable(triangle &&t)
      : id(hittable_id::Triangle), data(std::move(t)) {}
  __host__ __device__ inline hittable(bounding_box &&bb)
      : id(hittable_id::BoundingBox), data(std::move(bb)) {}
  __host__ __device__ inline hittable(bounding_array_node &&ban)
      : id(hittable_id::BoundingArrayNode), data(std::move(ban)) {}

  __host__ __device__ inline hittable() : id(hittable_id::Unknown), data() {}

  __host__ __device__ inline hittable(hittable const &h) : id(h.id), data() {
    switch (h.id) {
    case hittable_id::Sphere:
      data.s = h.data.s;
      break;
    case hittable_id::Plane:
      data.p = h.data.p;
      break;
    case hittable_id::Rectangle:
      data.r = h.data.r;
      break;
    case hittable_id::Triangle:
      data.t = h.data.t;
      break;
    case hittable_id::BoundingBox:
      data.bb = h.data.bb;
      break;
    case hittable_id::BoundingArrayNode:
      data.ban = h.data.ban;
      break;
    case hittable_id::Unknown:
      break;
    }
  }

  __host__ __device__ hittable &operator=(hittable const &h) {
    if (&h != this) {
      id = h.id;
      switch (h.id) {
      case hittable_id::Sphere:
        data.s = h.data.s;
        break;
      case hittable_id::Plane:
        data.p = h.data.p;
        break;
      case hittable_id::Rectangle:
        data.r = h.data.r;
        break;
      case hittable_id::Triangle:
        data.t = h.data.t;
        break;
      case hittable_id::BoundingBox:
        data.bb = h.data.bb;
        break;
      case hittable_id::BoundingArrayNode:
        data.ban = h.data.ban;
        break;
      case hittable_id::Unknown:
        break;
      }
    }
    return *this;
  }

  __host__ __device__ inline bool hit(const ray &r, num t_min, num t_max,
                                      hit_record &rec) const {
    return data.hit(id, r, t_min, t_max, rec);
  }

  __host__ __device__ inline bounding_box get_bounding_box() const {
    return data.get_bounding_box(id);
  }

  __host__ __device__ bounding_array_node const &
  as_bounding_array_node() const {
    return data.as_bounding_array_node();
  }

  hittable_id id;
  hittable_data data;
};

inline __host__ bounding_tree_node_node::bounding_tree_node_node(
    hittable **hittables, int hittables_size, RandomState *state, int start,
    int end) {

  hittable **temp_hittables = new hittable *[hittables_size];
  std::copy(hittables, hittables + hittables_size, temp_hittables);
  int range = end - start;

  int axis = random_int(state, 0, 2);
  auto comparator = (axis == 0)   ? x_compare
                    : (axis == 1) ? y_compare
                                  : z_compare;

  if (range == 1) {
    auto o = new bounding_tree_node_object();
    o->obj = temp_hittables[start];
    left = right = o;
  } else if (range == 2) {
    if (comparator(temp_hittables[start]->get_bounding_box(),
                   temp_hittables[start + 1]->get_bounding_box())) {
      auto o1 = new bounding_tree_node_object();
      o1->obj = temp_hittables[start];
      left = o1;
      auto o2 = new bounding_tree_node_object();
      o2->obj = temp_hittables[start + 1];
      right = o2;
    } else {
      auto o1 = new bounding_tree_node_object();
      o1->obj = temp_hittables[start + 1];
      left = o1;
      auto o2 = new bounding_tree_node_object();
      o2->obj = temp_hittables[start];
      right = o2;
    }
  } else {
    std::sort(temp_hittables + start, temp_hittables + end,
              [=](auto a, auto b) {
                return comparator(a->get_bounding_box(), b->get_bounding_box());
              });

    int mid = start + range / CONST(2);
    left = new bounding_tree_node_node(temp_hittables, hittables_size, state,
                                       start, mid);
    right = new bounding_tree_node_node(temp_hittables, hittables_size, state,
                                        mid, end);
  }

  delete[] temp_hittables;

  bounding_box box_left = left->get_bounding_box();
  bounding_box box_right = right->get_bounding_box();
  box = surrounding_box(box_left, box_right);
}

inline __host__ bounding_box
bounding_tree_node_object::get_bounding_box() const {
  return obj->get_bounding_box();
}

inline __host__ int convert_tree_to_array(bounding_tree_node *root,
                                          hittable *hittables) {
  static int node_index = 0;

  if (auto object = dynamic_cast<bounding_tree_node_object *>(root);
      object != nullptr) {
    return std::distance(hittables, object->obj);
  } else if (auto node = dynamic_cast<bounding_tree_node_node *>(root);
             node != nullptr) {
    if (node->left == node->right) {
      auto left_node = dynamic_cast<bounding_tree_node_object *>(node->left);
      hittables[node_index] = *left_node->obj;
      return std::distance(hittables, left_node->obj);
    } else {
      int index = node_index++;
      int left_id = convert_tree_to_array(node->left, hittables);
      int right_id = convert_tree_to_array(node->right, hittables);
      hittables[index] =
          bounding_array_node(left_id, right_id, root->get_bounding_box());
      return index;
    }
  }
  // Should definitely not get here
  return -1;
}

#endif