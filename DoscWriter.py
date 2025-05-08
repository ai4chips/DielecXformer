import os
import ast
import json
import subprocess
import sys

project_directory = os.path.dirname(os.path.abspath(__file__))

def find_code_files(directory):
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                code_files.append(os.path.join(root, file))
    return code_files

def extract_imports_from_py(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name.split('.')[0] for alias in node.names])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split('.')[0])
        return imports
    except Exception:
        return []

def extract_imports_from_ipynb(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        imports = []
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_code = ''.join(cell.get('source', ''))
                try:
                    tree = ast.parse(source_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            imports.extend([alias.name.split('.')[0] for alias in node.names])
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            imports.append(node.module.split('.')[0])
                except Exception:
                    continue
        return imports
    except Exception:
        return []


def get_all_imports(directory):
    all_imports = set()
    code_files = find_code_files(directory)
    for file in code_files:
        if file.endswith('.py'):
            all_imports.update(extract_imports_from_py(file))
        elif file.endswith('.ipynb'):
            all_imports.update(extract_imports_from_ipynb(file))
    return sorted(all_imports)


def get_package_version(package):
    try:
        result = subprocess.run(['pip', 'show', package], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith('Version:'):
                return line.split(':')[-1].strip()
    except Exception:
        return None

def get_all_versions(imports):
    versions = {}
    for pkg in imports:
        version = get_package_version(pkg)
        if version:
            versions[pkg] = version
    return versions


def write_requirements_md(package_versions, filename='requirements.md'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('# DieleXformer version 1.1.0\n')  # Add this line
        for pkg, ver in sorted(package_versions.items()):
            f.write(f'{pkg}=={ver}\n')




def write_installation_md(env_name, repo_url, package_versions, filename='installation.md'):
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    lines = [
        "# Installation",
        "",
        "Set up conda environment and clone the GitHub repo",
        "",
        "```bash",
        "# create a new environment",
        f"$ conda create --name {env_name} python={python_version}",
        f"$ conda activate {env_name}",
        "",
        "# install requirements",
    ]
    for pkg, ver in sorted(package_versions.items()):
        lines.append(f"$ pip install {pkg}=={ver}")
    lines.extend([
        "",
        f"# clone the source code of {env_name}",
        f"$ git clone {repo_url}",
        f"$ cd {env_name}",
        "```"
    ])
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))



if __name__ == '__main__':
    imports = get_all_imports(project_directory)
    package_versions = get_all_versions(imports)
    write_requirements_md(package_versions)

    write_installation_md(
        env_name='DieleXformer',
        repo_url='https://github.com/ai4chips/DieleXformer.git',
        package_versions=package_versions
    )
