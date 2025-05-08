import os

def find_large_files(directory, size_limit_mb=20):
    size_limit_bytes = size_limit_mb * 1024 * 1024
    large_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                if size > size_limit_bytes:
                    relative_path = os.path.relpath(filepath, directory)
                    large_files.append((relative_path, size / (1024 * 1024)))
            except FileNotFoundError:
                continue

    return large_files

if __name__ == "__main__":
    directory = "."
    result = find_large_files(directory)

    if result:
        print("Files larger than 20MB:")
        for path, size_mb in result:
            print(f"{path} - {size_mb:.2f} MB")
    else:
        print("No files larger than 20MB found.")
