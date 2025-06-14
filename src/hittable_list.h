#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "interval.h"
#include "aabb.h"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class hittable_list : public hittable
{
private:
    aabb box;
    hittable **list;
    int list_size;

public:
    std::vector<shared_ptr<hittable>> objects;

    __host__ __device__ hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    __device__ hittable_list(hittable **l, int n)
    {
        list = l;
        list_size = n;

        for(int i = 0; i < n; i++) {
            box = aabb(box, list[i]->bounding_box());
        }
    }

    __device__ int get_list_length() {
        return list_size;
    }

    void clear() { objects.clear(); }

    __host__ __device__ aabb bounding_box() const override
    {
        return box;
    }

    void add(shared_ptr<hittable> object)
    {
        objects.push_back(object);
        box = aabb(box, object->bounding_box());
    }

    //Todo: constructing box when device hittable list is done

    __host__ __device__ bool hit(const Ray &r, Interval interval, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = interval.max;

        for (const auto &object : objects)
        {
            if (object->hit(r, Interval(interval.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif