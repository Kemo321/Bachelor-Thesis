from pathlib import Path

def create_combined_cpp(src_dirs, include_dirs, config_patterns, output_filename, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {
            'build', 'out', 'bin', '.git', '.cache', 'venv', '__pycache__', 
            '.idea', '.cursor', 'external', 'extern', 'data', 'results', 'docs'
        }

    header_extensions = {'.hpp', '.h', '.cuh'}
    source_extensions = {'.cpp', '.c', '.cu'}
    root_dir = Path('.').resolve()

    def should_exclude(file_path):
        return any(excluded_folder in file_path.parts for excluded_folder in exclude_dirs)

    def get_files_by_ext(folders, extensions):
        found_files = []
        for folder in folders:
            path = Path(folder)
            if not path.exists():
                print(f"[WARNING] Folder {folder} does not exist. Skipping.")
                continue

            for ext in extensions:
                for match in path.rglob(f"*{ext}"):
                    if match.is_file() and not should_exclude(match):
                        found_files.append(match)

        return found_files

    def get_config_files(patterns, base_dir='.'):
        found_files = set()
        base_path = Path(base_dir)
        for pattern in patterns:
            for match in base_path.rglob(pattern):
                if match.is_file() and not should_exclude(match):
                    found_files.add(match)
        return sorted(list(found_files))

    headers = sorted(get_files_by_ext(include_dirs, header_extensions))
    sources = sorted(get_files_by_ext(src_dirs, source_extensions))
    configs = get_config_files(config_patterns)

    out_path = Path(output_filename).resolve()
    headers = [f for f in headers if f.resolve() != out_path]
    sources = [f for f in sources if f.resolve() != out_path]
    configs = [f for f in configs if f.resolve() != out_path]

    print(f"Found: {len(configs)} config/build files, {len(headers)} headers, and {len(sources)} source files.")

    def get_relative_path(file_path):
        try:
            return file_path.resolve().relative_to(root_dir)
        except ValueError:
            return file_path

    def write_section(outfile, files, section_title):
        if not files:
            return
        outfile.write("// " + "="*60 + "\n")
        outfile.write(f"// SECTION: {section_title}\n")
        outfile.write("// " + "="*60 + "\n\n")

        for file_path in files:
            rel_p = get_relative_path(file_path)
            outfile.write(f"// --- FILE: {rel_p} ---\n")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"// [ERROR READING FILE: {e}]\n")
            outfile.write("\n\n")

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write("// " + "="*60 + "\n")
        outfile.write("// AUTOMATICALLY GENERATED FILE - FULL PROJECT CONTEXT\n")
        outfile.write("// " + "="*60 + "\n\n")

        write_section(outfile, configs, "BUILD AND CONFIGURATION FILES")
        write_section(outfile, headers, "HEADER FILES")
        write_section(outfile, sources, "SOURCE FILES")

    print(f"Done! Full context written to '{output_filename}'.")


if __name__ == "__main__":
    INCLUDE_DIRECTORIES = ["./include"]
    SOURCE_DIRECTORIES = ["./src", "./benchmarks", "./tests"]

    EXCLUDE_DIRECTORIES = {
        'build', 'out', 'bin', '.git', '.cache', 'venv', '.vs', '.cursor', 
        'extern', 'external', 'data', 'results', 'docs', 'example', 'examples'
    }

    CONFIG_PATTERNS = [
        "CMakeLists.txt",
        "*.cmake",
        "Dockerfile*",
        "docker-compose*.yml",
        "docker-compose*.yaml",
        ".cursorrules",
        "*.sh"
    ]

    OUTPUT = "combined.cpp"

    create_combined_cpp(
        src_dirs=SOURCE_DIRECTORIES, 
        include_dirs=INCLUDE_DIRECTORIES, 
        config_patterns=CONFIG_PATTERNS, 
        output_filename=OUTPUT,
        exclude_dirs=EXCLUDE_DIRECTORIES
    )