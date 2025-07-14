#include "hittable_list_pod.cuh"
#include "hittable_pod.cuh"

// The main intersection logic for the list
__device__ bool hit_hittable_list(const HittableList *list, const Ray &r,
                                  Interval ray_t, HitRecord &rec)
{
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (int i = 0; i < list->size; i++)
    {
        Interval new_interval = {ray_t.min, closest_so_far};
        // Call the main hittable_hit dispatcher for each object in the list
        if (hittable_hit(list->objects[i], r, new_interval, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// The bounding box function simply returns the pre-computed box
__device__ aabb bounding_box_hittable_list(const HittableList *list)
{
    return list->bbox;
}