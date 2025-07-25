#include "bvh_node_pod.cuh"

namespace cuda_device
{
    __device__ inline Hittable fetch_hittable(cudaTextureObject_t texture, int index)
    {
        // A 64-byte Hittable is exactly four float4's.
        int base_idx = index * 4;

        // --- Create a temporary, local array to hold the raw data ---
        float4 data[4];

        // --- Fetch all four 16-byte chunks that make up the Hittable ---
        data[0] = tex1Dfetch<float4>(texture, base_idx + 0);
        data[1] = tex1Dfetch<float4>(texture, base_idx + 1);
        data[2] = tex1Dfetch<float4>(texture, base_idx + 2);
        data[3] = tex1Dfetch<float4>(texture, base_idx + 3);

        // --- Reinterpret the raw data as a Hittable object ---
        // This is now safe and extremely fast because you've guaranteed
        // that the data layout and alignment are correct.
        return *(reinterpret_cast<Hittable *>(data));
    }

    __device__ bool hit_bvh(const BVHNode *bvh_nodes,
                            cudaTextureObject_t objects_texture, int node_idx,
                            const Ray &r, Interval ray_t, HitRecord &rec)
    {
        bool hit_anything = false;
        int to_visit_stack[MAX_NODE_LEVEL];
        int stack_idx = 0;
        to_visit_stack[stack_idx++] = node_idx;

        while (stack_idx > 0)
        {
            int current_node_idx = to_visit_stack[--stack_idx];
            const BVHNode &node = bvh_nodes[current_node_idx];

            // First, check if the ray hits the current node's bounding box
            if (!aabb_hit(node.bbox, r, ray_t))
            {
                continue;
            }

            // If it's a leaf node, check for intersection with the object
            if (node.is_leaf)
            {
                int start_idx = node.left_child_idx;
                int count = node.right_child_idx;

                for (int i = 0; i < count; ++i)
                {
                    const Hittable &obj = fetch_hittable(objects_texture, start_idx + i);
                    if (hittable_hit(obj, r, ray_t, rec))
                    {
                        hit_anything = true;
                        ray_t.max = rec.t;
                    }
                }
            }
            else
            {
                // If it's an internal node, push children onto the stack
                // A simple optimization could be to check which child is closer first
                to_visit_stack[stack_idx++] = node.left_child_idx;
                to_visit_stack[stack_idx++] = node.right_child_idx;
            }
        }

        return hit_anything;
    }
}