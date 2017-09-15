set(TARGET_NAME image)
setup_hifi_library()
link_hifi_libraries(shared gpu)

add_dependency_external_projects(tbb)

target_glm()

add_dependency_external_projects(nvtt)
find_package(NVTT REQUIRED)
find_package(TBB REQUIRED)

target_include_directories(${TARGET_NAME} PRIVATE ${NVTT_INCLUDE_DIRS})
# append tbb includes to our list of includes to bubble
target_include_directories(${TARGET_NAME} PRIVATE ${TBB_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} ${NVTT_LIBRARIES} ${TBB_LIBRARIES})
add_paths_to_fixup_libs(${NVTT_DLL_PATH})
