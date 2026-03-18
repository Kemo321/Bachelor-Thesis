# Format code

clang-format -i -style=file $(find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.c" -o -name "*.hpp" \))

# Run pre-commit

pre-commit run --all-files
