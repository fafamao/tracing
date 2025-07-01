#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "interval.h"
#include "aabb.h"

#include <memory>
#include <vector>

class hittable_list : public hittable
{
private:
    aabb box;
    hittable **list;
    int list_size;

public:
    std::vector<hittable *> objects;
    __host__ __device__ hittable_list() {}
    hittable_list(hittable *object_ptr) { add(object_ptr); }

    __device__ hittable_list(hittable **l, int n)
    {
        list = l;
        list_size = n;

        // TODO: validate this.
        for (int i = 0; i < n; i++)
        {
            box = aabb(box, list[i]->bounding_box());
        }
    }

    __device__ int get_list_length()
    {
        return list_size;
    }

    void clear() { objects.clear(); }

    __host__ __device__ aabb bounding_box() const override
    {
        return box;
    }

    void add(hittable *object)
    {
        objects.push_back(object);
        box = aabb(box, object->bounding_box());
    }

    __host__ __device__ bool hit(const Ray &r, Interval interval, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = interval.max;

#ifdef __CUDA_ARCH__
        for (int i = 0; i < list_size; i++)
        {
            if (list[i]->hit(r, Interval(interval.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
#else
        for (const auto &object : objects)
        {
            if (object->hit(r, Interval(interval.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
#endif
        return hit_anything;
    }
};

#endif