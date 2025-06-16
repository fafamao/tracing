#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

#include <algorithm>

class bvh_node : public hittable
{
public:
    bvh_node(hittable_list list) : bvh_node(list.objects, 0, list.objects.size())
    {
        // TODO: remove copy
    }

    bvh_node(std::vector<shared_ptr<hittable>> &objects, size_t start, size_t end)
    {
        // TODO: fix random number generator
        int axis = random_int(0, 2);

        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1)
        {
            left_ptr = right_ptr = objects[start].get();
        }
        else if (object_span == 2)
        {
            left_ptr = objects[start].get();
            right_ptr = objects[start + 1].get();
        }
        else
        {
            std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

            auto mid = start + object_span / 2;
            left_ptr = new bvh_node(objects, start, mid);
            right_ptr = new bvh_node(objects, mid, end);
        }

        bbox = aabb(left->bounding_box(), right->bounding_box());
    }
    __device__ bvh_node(hittable** objects, size_t start, size_t end)
    {
        // TODO: fix random number generator
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

        bbox = aabb(left->bounding_box(), right->bounding_box());
    }

    bool hit(const Ray &r, Interval ray_t, hit_record &rec) const override
    {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, Interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

private:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    hittable *left_ptr;
    hittable *right_ptr;
    aabb bbox;

    __host__ __device__ static bool box_compare(
        const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis_index)
    {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    __host__ __device__ static bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b)
    {
        return box_compare(a, b, 0);
    }

    __host__ __device__ static bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b)
    {
        return box_compare(a, b, 1);
    }

    __host__ __device__ static bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b)
    {
        return box_compare(a, b, 2);
    }
};

#endif // BVH_NODE_H