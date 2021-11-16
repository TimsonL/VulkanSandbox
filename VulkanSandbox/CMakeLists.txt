cmake_minimum_required(VERSION 3.5)

# set the project name
project(VulkanSandbox VERSION 1.0)

set(SOURCES
    main.cpp
    Application.cpp
)
set(HEADERS
    Application.h
)

set(VULKAN_SDK_PATH
    $ENV{VULKAN_SDK}
)

# add the executable
add_executable(VulkanSandbox ${SOURCES} ${HEADERS})

set_target_properties(
    VulkanSandbox PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(
    VulkanSandbox PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/glfw-3.3.2.bin.WIN64/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/glm
    ${VULKAN_SDK_PATH}/Include
)

find_package(Vulkan REQUIRED FATAL_ERROR)

target_link_libraries(
    VulkanSandbox PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/glfw-3.3.2.bin.WIN64/lib-vc2019/glfw3.lib
    ${VULKAN_SDK_PATH}/Lib/vulkan-1.lib
)