from pathlib import Path

def create_combined_cpp(src_dirs, include_dirs, output_filename):
    header_extensions = {'.hpp', '.h'}
    source_extensions = {'.cpp'}

    def get_files(folders, extensions):
        found_files = []
        for folder in folders:
            path = Path(folder)
            if not path.exists():
                print(f"[WARNING] Folder {folder} does not exist. Skipping.")
                continue

            for ext in extensions:
                found_files.extend(path.rglob(f"*{ext}"))

        return sorted(found_files)

    headers = get_files(include_dirs, header_extensions)
    sources = get_files(src_dirs, source_extensions)

    print(f"Found {len(headers)} headers and {len(sources)} source files.")

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write("// " + "="*60 + "\n")
        outfile.write("// AUTOMATICALLY GENERATED FILE - ALL MODULES\n")
        outfile.write("// " + "="*60 + "\n\n")

        for header in headers:
            outfile.write(f"// --- FILE: {header.relative_to(Path('.').absolute().parent if header.is_absolute() else '.')} ---\n")
            with open(header, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write("\n\n")

        for source in sources:
            if source.name == output_filename:
                continue

            outfile.write(f"// --- FILE: {source.relative_to(Path('.').absolute().parent if source.is_absolute() else '.')} ---\n")
            with open(source, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write("\n\n")

    print(f"Done! Everything was written to '{output_filename}'.")

if __name__ == "__main__":
    INCLUDE_DIRECTORIES = ["./include/DeepLearnLib"]
    SOURCE_DIRECTORIES = ["./src", "./benchmarks"]
    OUTPUT = "combined.cpp"

    create_combined_cpp(SOURCE_DIRECTORIES, INCLUDE_DIRECTORIES, OUTPUT)