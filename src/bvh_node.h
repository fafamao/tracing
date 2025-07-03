#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

#include <algorithm>
#include <thrust/sort.h>

class bvh_node : public hittable
{
public:
    bvh_node(hittable_list list) : bvh_node(list.objects, 0, list.objects.size())
    {
        // TODO: remove copy
    }
    __device__ bvh_node(hittable **list, size_t list_length, curandState &rand_state) : bvh_node(list, 0, list_length, rand_state)
    {
        // TODO: remove copy
    }

    bvh_node(std::vector<hittable *> objects, size_t start, size_t end)
    {
        int axis = random_int(0, 2);

        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1)
        {
            left_ptr = right_ptr = objects[start];
        }
        else if (object_span == 2)
        {
            left_ptr = objects[start];
            right_ptr = objects[start + 1];
        }
        else
        {
            std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

            auto mid = start + object_span / 2;
            left_ptr = new bvh_node(objects, start, mid);
            right_ptr = new bvh_node(objects, mid, end);
        }

        bbox = aabb(left_ptr->bounding_box(), right_ptr->bounding_box());
    }
    __device__ bvh_node(hittable **objects, size_t start, size_t end, curandState &rand_state)
    {
        // TODO: validate this.
        int axis = (int)(curand_uniform(&rand_state) * 3.0f);

        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1)
        {
            left_ptr = right_ptr = objects[start];
        }
        else if (object_span == 2)
        {
            left_ptr = objects[start];
            right_ptr = objects[start + 1];
        }
        else
        {
            thrust::sort(objects + start, objects + end, comparator);

            auto mid = start + object_span / 2;
            left_ptr = new bvh_node(objects, start, mid, rand_state);
            right_ptr = new bvh_node(objects, mid, end, rand_state);
        }

        bbox = aabb(left_ptr->bounding_box(), right_ptr->bounding_box());
    }

    __device__ __host__ bool hit(const Ray &r, Interval ray_t, hit_record &rec) const override
    {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left_ptr->hit(r, ray_t, rec);
        bool hit_right = right_ptr->hit(r, Interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

    ~bvh_node()
    {
        delete left_ptr;
        delete right_ptr;
    }

private:
    hittable *left_ptr;
    hittable *right_ptr;
    aabb bbox;

    __host__ __device__ static bool box_compare(
        const hittable *a, const hittable *b, int axis_index)
    {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    __host__ __device__ static bool box_x_compare(const hittable *a, const hittable *b)
    {
        return box_compare(a, b, 0);
    }

    __host__ __device__ static bool box_y_compare(const hittable *a, const hittable *b)
    {
        return box_compare(a, b, 1);
    }

    __host__ __device__ static bool box_z_compare(const hittable *a, const hittable *b)
    {
        return box_compare(a, b, 2);
    }
};

#endif // BVH_NODE_H