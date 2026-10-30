option(EDGEINFER_PROFILE "Build per-operator profiler instead of whole-network benchmark" OFF)

if(EDGEINFER_PROFILE)
    set(app_source "main_profile.cpp")
else()
    set(app_source "main.cpp")
endif()

idf_component_register(
    SRCS "${app_source}"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp-tflite-micro freertos esp_timer
)
