add_library(TorchBaseline TorchYOLO.cpp TorchDataset.cpp)

target_include_directories(
  TorchBaseline
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE "${CMAKE_SOURCE_DIR}/extern/stb")

target_include_directories(TorchBaseline SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(TorchBaseline PUBLIC DeepLearnLib ${TORCH_LIBRARIES})

set_target_properties(TorchBaseline PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

if(MSVC)
  target_compile_options(TorchBaseline PRIVATE /bigobj /Zc:preprocessor)
endif()
