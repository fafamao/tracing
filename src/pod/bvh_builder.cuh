#ifndef POD_BVH_BUILDER_CUH_
#define POD_BVH_BUILDER_CUH_

#include "aabb_pod.cuh"
#include "hittable_pod.cuh"

#include <vector>
#include <algorithm>
#include <iostream>

#include "hittable_pod.cuh"
#include "bvh_node_pod.cuh"
#include "aabb_pod.cuh"

// A helper function to get the bounding box from a Hittable.
// This would call your main hittable_bounding_box dispatcher.
inline aabb get_hittable_bounding_box(const Hittable &object)
{
    switch (object.type)
    {
    case SPHERE:
        return bounding_box_sphere(object.sphere);
    // TODO: Add other object types here
    default:
        return aabb(); // Return empty AABB
    }
}

class BVHBuilder
{
public:
    BVHBuilder(std::vector<Hittable> &scene_objects)
        : objects(scene_objects) {}

    std::vector<BVHNode> build()
    {
        if (objects.empty())
        {
            return {};
        }

        std::cout << "Starting BVH build for " << objects.size() << " objects." << std::endl;

        // The root node is at index 0.
        build_recursive(0, objects.size());

        std::cout << "BVH build complete. Total nodes: " << nodes.size() << std::endl;
        return nodes;
    }

private:
    std::vector<Hittable> &objects;
    std::vector<BVHNode> nodes;

    // The core recursive function to build the BVH tree.
    // It takes a range [start, end) of the objects vector.
    // It returns the index of the node it created in the `nodes` array.
    int build_recursive(size_t start, size_t end)
    {
        // Create a new node and add it to our list.
        int node_idx = nodes.size();
        // Add a default-constructed node
        nodes.emplace_back();

        size_t span = end - start;

        // Base Case: If the range has few enough objects, create a leaf node.
        if (span <= 4)
        { // Leaf node condition (can be tuned)
            nodes[node_idx].is_leaf = true;
            // For a leaf, the 'left_child_idx' can store the start of the object range,
            // and 'right_child_idx' can store the count.
            nodes[node_idx].left_child_idx = start;
            nodes[node_idx].right_child_idx = span;

            // Compute the bounding box for all objects in this leaf.
            aabb leaf_box = get_hittable_bounding_box(objects[start]);
            for (size_t i = start + 1; i < end; ++i)
            {
                leaf_box = create_aabb_from_boxes(leaf_box, get_hittable_bounding_box(objects[i]));
            }
            nodes[node_idx].bbox = leaf_box;

            return node_idx;
        }

        // Recursive Step: It's an internal node.
        nodes[node_idx].is_leaf = false;

        // 1. Choose a random axis to split on.
        // 0=x, 1=y, 2=z
        int axis = random_int(0, 2);

        // 2. Sort the objects in the range [start, end) along the chosen axis.
        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;
        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        // 3. Find the midpoint and recursively call for left and right children.
        size_t mid = start + span / 2;
        int left_child = build_recursive(start, mid);
        int right_child = build_recursive(mid, end);

        // 4. Set the children and bounding box for the current internal node.
        nodes[node_idx].left_child_idx = left_child;
        nodes[node_idx].right_child_idx = right_child;
        nodes[node_idx].bbox = create_aabb_from_boxes(nodes[left_child].bbox, nodes[right_child].bbox);

        return node_idx;
    }

    static bool box_compare(const Hittable &a, const Hittable &b, int axis_index)
    {
        aabb box_a = get_hittable_bounding_box(a);
        aabb box_b = get_hittable_bounding_box(b);
        const Interval &a_interval = aabb_axis_interval(box_a, axis_index);
        const Interval &b_interval = aabb_axis_interval(box_b, axis_index);
        return a_interval.min < b_interval.min;
    }

    static bool box_x_compare(const Hittable &a, const Hittable &b)
    {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare(const Hittable &a, const Hittable &b)
    {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare(const Hittable &a, const Hittable &b)
    {
        return box_compare(a, b, 2);
    }
};

#endif // POD_BVH_BUILDER_CUH_