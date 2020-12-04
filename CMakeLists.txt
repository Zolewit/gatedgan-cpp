cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(gatedgan)

set(CMAKE_PREFIX_PATH "C:\\Users\\User\\libtorch\\debug;C:\\Users\\User\\opencv\\build\\bin")
cmake_policy(SET CMP0054 NEW)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(${PROJECT_NAME} data.h models.h utils.h builder.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)