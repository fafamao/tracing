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
namespace cuda_device
{
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

        void print_tree() const
        {
            std::cout << "\n--- BVH Tree Structure ---\n";
            if (!nodes.empty())
            {
                // Start the recursive printing from the root node (index 0)
                print_node(0, 0);
            }
            std::cout << "--- End of BVH Tree ---\n"
                      << std::endl;
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
            {
                nodes[node_idx].is_leaf = true;
                // For a leaf, the 'left_child_idx' can store the start of the object range,
                // and 'right_child_idx' can store the count.
                nodes[node_idx].left_child_idx = start;
                nodes[node_idx].right_child_idx = span;

                // Compute the bounding box for all objects in this leaf.
                aabb leaf_box = hittable_bounding_box(objects[start]);
                for (size_t i = start + 1; i < end; ++i)
                {
                    leaf_box = create_aabb_from_boxes(leaf_box, hittable_bounding_box(objects[i]));
                }
                nodes[node_idx].bbox = leaf_box;

                return node_idx;
            }

            // Recursive Step: It's an internal node.
            nodes[node_idx].is_leaf = false;

            int axis = random_int(0, 2);

            auto comparator = (axis == 0)   ? box_x_compare
                              : (axis == 1) ? box_y_compare
                                            : box_z_compare;
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            size_t mid = start + span / 2;
            int left_child = build_recursive(start, mid);
            int right_child = build_recursive(mid, end);

            // 4. Set the children and bounding box for the current internal node.
            nodes[node_idx].left_child_idx = left_child;
            nodes[node_idx].right_child_idx = right_child;
            nodes[node_idx].bbox = create_aabb_from_boxes(nodes[left_child].bbox, nodes[right_child].bbox);

            return node_idx;
        }

        void print_node(int node_idx, int depth) const
        {
            // Create an indentation string based on the tree depth
            std::string indent(depth * 4, ' ');
            indent += "|-- ";

            const BVHNode &node = nodes[node_idx];

            // Print the current node's information
            std::cout << indent << "Node " << node_idx << " " << node.bbox << std::endl;

            // Check if it's a leaf or an internal node
            if (node.is_leaf)
            {
                int start_idx = node.left_child_idx;
                int num_objects = node.right_child_idx;
                std::cout << indent << "    (Leaf) Contains " << num_objects << " object(s) starting at index " << start_idx << ":" << std::endl;

                // Print details for each object in the leaf
                for (int i = 0; i < num_objects; ++i)
                {
                    const Hittable &obj = objects[start_idx + i];
                    // Assuming Sphere is the only object type for now
                    const Sphere &sphere = obj.sphere;
                    std::cout << indent << "      - " << object_type_to_string(obj.type)
                              << " at " << sphere.center0
                              << ", r=" << sphere.radius << std::endl;
                }
            }
            else
            {
                // It's an internal node, so recurse for its children
                std::cout << indent << "    (Internal) Children: " << node.left_child_idx << " and " << node.right_child_idx << std::endl;
                print_node(node.left_child_idx, depth + 1);
                print_node(node.right_child_idx, depth + 1);
            }
        }

        static bool box_compare(const Hittable &a, const Hittable &b, int axis_index)
        {
            aabb box_a = hittable_bounding_box(a);
            aabb box_b = hittable_bounding_box(b);
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
}
#endif // POD_BVH_BUILDER_CUH_