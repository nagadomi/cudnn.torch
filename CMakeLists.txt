
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

IF(LUAROCKS_PREFIX)
  MESSAGE(STATUS "Installing CuDNN through Luarocks")
  STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
  MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 8.0 REQUIRED)

#FIND_PACKAGE(CUDNN 6 EXACT QUIET)
#IF(NOT CUDNN_FOUND)
#  CUDNN_INSTALL(6.0-rc "${Torch_INSTALL_LIB}" "${Torch_INSTALL_INCLUDE}" "")
#  FIND_PACKAGE(CUDNN 6 EXACT REQUIRED)
#ENDIF()


FILE(GLOB luasrc *.lua)
SET(src "")
ADD_TORCH_PACKAGE(cudnn "${src}" "${luasrc}" "NVIDIA CuDNN Bindings")
