| Tag | Purpose | Example |
|-----|---------|---------|
| feat | A new feature for the user | feat: add ReLU activation function |
| fix | A bug fix | fix: resolve memory leak in Tensor destructor |
| docs | Documentation only changes | docs: update build instructions in README |
| style | Formatting, missing semi-colons, etc. (no logic change) | style: run clang-format on src/ |
| refactor | Code change that neither fixes a bug nor adds a feature | refactor: simplify matrix multiplication loop |
| perf | A code change that improves performance | perf: use AVX instructions for dot product |
| test | Adding missing tests or correcting existing tests | test: add unit tests for backpropagation |
| build | Changes to the build system or external dependencies | build: upgrade CMake version to 3.20 |
| ci | Changes to CI configuration files and scripts | ci: add gcc-15 to github actions matrix |
| chore | Other changes that don't modify src or test files | chore: add .gitignore |
| revert | Reverts a previous commit | revert: feat: add experimental cuda support |