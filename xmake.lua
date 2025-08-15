--[[
xmake.lua
This is the build script for the 'tracing' project.
]]

-- == Global Settings ==

-- Set the C++ standard globally for the entire project.
set_languages("c++17")

-- Add the compiler flag to show CUDA resource usage upon compilation.
add_cuflags("--ptxas-options=-v")

-- Define required packages
add_requires("cuda")

-- Define a workspace-level rule for CUDA toolkit paths
add_includedirs("/usr/local/cuda/include", { public = true })
add_linkdirs("/usr/local/cuda/lib64")

-- Define common include paths for all targets
add_includedirs("src", "utility")

-- == Build Mode Configurations ==

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
    set_strip("all")
    add_cuflags("--ptxas-options=--maxrregcount=60")
elseif is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
elseif is_mode("profile") then
    set_symbols("debug")
    add_cxflags("-pg")
    add_ldflags("-pg")
end

-- == Library Targets ==

target("camera_library")
    set_kind("static")
    add_files("src/legacy/camera.cu")

target("scene_library")
    set_kind("static")
    add_files("src/legacy/scene.cu")

target("rng_library")
    set_kind("static")
    add_files("src/random_number_generator.cu")

target("render_library")
    set_kind("static")
    add_files("src/legacy/render.cu")

target("ray_tracing_lib")
    set_kind("static")
    add_files("src/pod/*.cc", "src/pod/*.cu")

-- == Main Executable Target ==

target("tracing")
    -- Set the kind to a binary executable
    set_kind("binary")

    -- Add the main source file
    add_files("main.cu")

    -- Add dependencies on our static libraries
    add_deps("camera_library", "scene_library", "rng_library", "render_library", "ray_tracing_lib")

    -- Add package dependencies
    add_packages("cuda")

    -- Set the output directory and filename
    set_targetdir("bin")
    set_basename("tracing")
