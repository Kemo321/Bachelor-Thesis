file(GLOB SOURCES "*.cpp")

# Define the library target
add_library(${PROJECT_NAME} ${SOURCES})

# Specify where headers are located
target_include_directories(${PROJECT_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)
