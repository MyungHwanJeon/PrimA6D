cmake_minimum_required(VERSION 2.8.12)
project(bop_renderer)


################################################################################
# Before running cmake, set the following environment variables:

# OSMESA_PREFIX - Folder with the OSMesa installation
# PYTHON_PREFIX - Folder with "lib" and "include" subfolders with Python sources

# Example:
# export OSMESA_PREFIX=/opt/osmesa
# export PYTHON_PREFIX=/usr
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

set(OSMESA_PREFIX /opt/osmesa)
set(PYTHON_PREFIX /usr)

# Enable support for C++11.
set(CPP_STANDARD -std=c++11)
add_definitions(${CPP_STANDARD})

# Python library (tested with Python 3.6).
set(PYTHON_LIBRARIES ${PYTHON_PREFIX}/lib/libpython3.6m.so)

# Directory with Python includes (tested with Python 3.6).
set(PYTHON_INCLUDE_DIRS ${PYTHON_PREFIX}/include/python3.6m)
include_directories(${PYTHON_PREFIX}/include)

# OSMesa.
set(OSMESA_LIBRARIES $ENV{OSMESA_PREFIX}/lib/libOSMesa32.so)
include_directories($ENV{OSMESA_PREFIX}/include)

# Pybind11.
set(PYBIND11_CPP_STANDARD ${CPP_STANDARD})
set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(3rd/pybind11)

# Others.

include_directories(src)
include_directories(3rd)
include_directories(3rd/glm)

# Header files.
set(HEADERFILES
        src/Renderer.h
        src/PythonWrapper.h
        src/AbstractLoader.h
        src/PlyLoader.h
        src/GPUBuffer.h
        src/Pose.h
        src/Geometry.h
        src/glutils/FrameBufferObject.h
        src/glutils/GLSLProgram.h
        src/glutils/gl_core_3_3.h
        src/context/Context.h
        src/Texture.h
        src/Shaders.h
        )

# Source files.
set(SOURCEFILES
        src/Renderer.cpp
        src/AbstractLoader.cpp
        src/PlyLoader.cpp
        src/GPUBuffer.cpp
        src/Pose.cpp
        src/glutils/FrameBufferObject.cpp
        src/glutils/GLSLProgram.cpp
        src/glutils/gl_core_3_3.cpp
        src/context/OSMContext.cpp
        3rd/rply/rply.c
        3rd/lodepng/lodepng.cpp
        )

#add_library(
#	bop_renderer
#	SHARED
#	${SOURCEFILES}
#	)

# Python module.
pybind11_add_module(bop_renderer MODULE
        src/PythonWrapper.cpp ${HEADERFILES} ${SOURCEFILES})
target_link_libraries(bop_renderer PRIVATE ${OSMESA_LIBRARIES})
