add_requires("cuda")

if is_mode("profile") then
    set_symbols("debug")
    add_cxflags("-pg")
    add_ldflags("-pg")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
    set_strip("all")
end

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

target("cuda")
    set_kind("phony")
    add_includedirs("/usr/local/cuda-12.9/include")
    add_linkdirs("/usr/local/cuda-12.9/lib64")
    add_links("cudart")

target("camera_library")
    set_kind("static")
    add_includedirs("src")
    add_includedirs("utility")
    add_files("src/camera.cu")

target("tracing")
    set_kind("binary")
    add_includedirs("src")
    add_includedirs("utility")
    add_files("main.cu")
    add_deps("camera_library")
    add_deps("cuda")
    set_toolchains("cuda")
    set_targetdir("bin")
    set_basename("tracing")
