#ifndef HITTABLE_LIST_POD_CUH_
#define HITTABLE_LIST_POD_CUH_

#include "hittable_pod.cuh"
#include "aabb_pod.cuh"

struct HittableList
{
    Hittable *objects; // Pointer to the array of hittable objects on the device
    int size;
    aabb bbox; // The bounding box of the entire list
};

// The main intersection logic for the list
__device__ inline bool hit_hittable_list(
    const HittableList *list, const Ray &r, Interval ray_t, HitRecord &rec)
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
__device__ inline aabb bounding_box_hittable_list(const HittableList *list)
{
    return list->bbox;
}

#endif // HITTABLE_LIST_POD_CUH_