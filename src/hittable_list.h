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

public:
    std::vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    aabb bounding_box() const override
    {
        return box;
    }

    void add(shared_ptr<hittable> object)
    {
        objects.push_back(object);
        box = aabb(box, object->bounding_box());
    }

    bool hit(const Ray &r, Interval &interval, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = interval.max;

        for (const auto &object : objects)
        {
            auto new_interval = Interval(interval.min, closest_so_far);
            if (object->hit(r, new_interval, temp_rec))
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