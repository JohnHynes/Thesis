#ifndef GPU_RAY_TRACER_HITTABLE_HPP_
#define GPU_RAY_TRACER_HITTABLE_HPP_

#include "Preprocessor.hpp"
#include "Ray.hpp"
#include "constants.hpp"
#include "types.hpp"
#include <glm/geometric.hpp>
#include <limits>

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
  BoundingNode,
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

struct bounding_tree_node : public IHittable {
public:
  // positive idx = node, negative idx = leaf
  bounding_tree_node* left, right;
  bounding_box box;

public:
  __host__ __device__ bounding_tree_node(bounding_tree_node* l, bounding_tree_node* r, bounding_box b)
      : left(l), right(r), box(b) {}

  __host__ bounding_box surrounding_box(bounding_box b1, bounding_box b2)
  {
    point3 min, max;
    for (int i = 0; i < 3; ++i) {
      min[i] = std::min(b1.minimum[i], b2.minimum[i]);
      max[i] = std::max(b1.maximum[i], b2.maximum[i]);
    }
    return bounding_box(min, max);
  }

  template<typename HittableArray>
  __host__ bounding_tree_node(const HittableArray src_objects, int objects_count, int start, int end, double time0, double time1)
  {
    HittableArray temp_objects[objects_count];
    std::copy(src_objects, src_objects + objects_count, temp_objects);
    int range = end - start;

    auto comparator; // make comparison function

    if (range == 1)
    {
      left = right = objects[start];
    }
    else if (range == 2)
    {
      if (comparator(temp_objects[start], temp_objects[start + 1])) {
          left = temp_objects[start];
          right = temp_objects[start + 1];
      } else {
          left = temp_objects[start + 1];
          right = temp_objects[start];
      }
    } else {
        std::sort(temp_objects + start, temp_objects + end, comparator);

        int mid = start + range / CONST(2);
        left = &bounding_tree_node(temp_objects, object_count, start, mid, time0, time1);
        right = &bounding_tree_node(temp_objects, objects_count, mid, end, time0, time1);
    }

    bounding_box box_left = left->bounding_box();
    bounding_box box_right = right->bounding_box();
    box = surrounding_box(box_left, box_right);
  }

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
    // 1. Use the Moller-Trumbore ray-triangle algorithm to compute t
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
  bounding_node bn;
  null_hittable n;

  __host__ __device__ inline hittable_data(sphere const &s) : s(s) {}
  __host__ __device__ inline hittable_data(plane const &p) : p(p) {}
  __host__ __device__ inline hittable_data(rectangle const &r) : r(r) {}
  __host__ __device__ inline hittable_data(triangle const &t) : t(t) {}
  __host__ __device__ inline hittable_data(bounding_box const &bb) : bb(bb) {}
  __host__ __device__ inline hittable_data(bounding_node const &bn) : bn(bn) {}

  __host__ __device__ inline hittable_data(sphere &&s) : s(std::move(s)) {}
  __host__ __device__ inline hittable_data(plane &&p) : p(std::move(p)) {}
  __host__ __device__ inline hittable_data(rectangle &&r) : r(std::move(r)) {}
  __host__ __device__ inline hittable_data(triangle &&t) : t(std::move(t)) {}
  __host__ __device__ inline hittable_data(bounding_box &&bb)
      : bb(std::move(bb)) {}
  __host__ __device__ inline hittable_data(bounding_node &&bn)
      : bn(std::move(bn)) {}

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
    case hittable_id::BoundingNode:
      return bn.hit(std::forward<Args>(args)...);
    default:
      return n.hit(std::forward<Args>(args)...);
    }
  }

  __host__ __device__ inline bounding_box get_bounding_box(hittable_id id) const {
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
    case hittable_id::BoundingNode:
      return bn.get_bounding_box();
    default:
      return n.get_bounding_box();
    }
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
  __host__ __device__ inline hittable(bounding_node const &bn)
      : id(hittable_id::BoundingNode), data(bn) {}

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
  __host__ __device__ inline hittable(bounding_node &&bn)
      : id(hittable_id::BoundingNode), data(std::move(bn)) {}

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
    case hittable_id::BoundingNode:
      data.bn = h.data.bn;
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
      case hittable_id::BoundingNode:
        data.bn = h.data.bn;
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

  hittable_id id;
  hittable_data data;
};

#endif