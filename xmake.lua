if is_mode("profile") then
    set_symbols("debug")
    add_cxflags("-pg")
    add_ldflags("-pg")
end
target("camera_library")
    set_kind("static")
    add_includedirs("src")
    add_includedirs("utility")
    add_files("src/camera.cpp")

target("interval_library")
    set_kind("static")
    add_includedirs("src")
    add_files("src/interval.cpp")

target("tracing")
    set_kind("binary")
    add_includedirs("src")
    add_includedirs("utility")
    add_files("main.cpp")
    add_deps("camera_library")