set(TARGET_NAME graphics)
AUTOSCRIBE_SHADER_LIB(gpu graphics render-utils)
setup_hifi_library()
link_hifi_libraries(shared ktx gpu image)