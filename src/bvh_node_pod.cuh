#ifndef BVH_NODE_POD_CUH_
#define BVH_NODE_POD_CUH_

#include "hittable_pod.cuh"
#include "aabb_pod.cuh"

/* Take std::vector of Hittable PODs.

Recursively subdivide the list and sort it, just like before.

Instead of calling new bvh_node(...), it will populate the flat BVHNode array. It will manage the indices for parent and child nodes.

Once the host-side array of BVHNodes is complete, use cudaMalloc and cudaMemcpy to copy the entire flat structure to the GPU. */

struct BVHNode
{
    aabb bbox;
    int left_child_idx;  // Index into the BVH node array
    int right_child_idx; // Index into the BVH node array

    // For leaf nodes
    bool is_leaf;
    int object_idx; // Index into the main hittable object array
};

__device__ inline bool hit_bvh(
    const BVHNode *bvh_nodes, const Hittable *objects, // Pass in the arrays
    int node_idx, const Ray &r, Interval ray_t, HitRecord &rec)
{
    // --- Iterative Traversal (Replaces Recursion) ---
    bool hit_anything = false;
    // TODO: dynamic stack
    int to_visit_stack[64]; // A fixed-size stack for traversal
    int stack_idx = 0;
    to_visit_stack[stack_idx++] = node_idx; // Start with the root node

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
            if (hittable_hit(objects[node.object_idx], r, ray_t, rec))
            {
                hit_anything = true;
                // Narrow the interval to find the closest hit
                ray_t.max = rec.t;
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

#endif // BVH_NODE_POD_CUH_