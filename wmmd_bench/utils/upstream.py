"""WMMD-authored external checkout validation and import boundary.

Validation reads metadata and hashes only. It never imports upstream code.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def contracts():
    return json.loads((ROOT / 'configs/evaluation/upstream.json').read_text(encoding='utf-8'))['methods']


def validate(method, root=None):
    contract = contracts()[method]
    chosen = root or os.environ.get('WMMD_UPSTREAM_ROOT')
    if not chosen:
        raise ValueError('WMMD_UPSTREAM_ROOT is required; no checkout is downloaded automatically')
    checkout = Path(chosen).resolve() / method
    if not checkout.is_dir():
        raise ValueError(f'Missing upstream checkout for {method}')
    def git(*args):
        result = subprocess.run(['git', '-C', str(checkout), *args], capture_output=True,
                                text=True, timeout=15, check=False)
        if result.returncode:
            raise ValueError(f'{method}: cannot verify Git checkout identity')
        return result.stdout.strip()
    # A directory inside the benchmark's own Git repo must not pass as a checkout.
    if Path(git('rev-parse', '--show-toplevel')).resolve() != checkout.resolve():
        raise ValueError(f'{method}: expected a separate upstream checkout')
    if git('rev-parse', 'HEAD') != contract['revision']:
        raise ValueError(f'{method}: wrong revision; expected {contract["revision"]}')
    checked = []
    for relative, expected in contract['required_files'].items():
        file = checkout / relative
        if not file.is_file() or not file.resolve().is_relative_to(checkout.resolve()):
            raise ValueError(f'{method}: required upstream file missing or outside checkout: {relative}')
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{method}: upstream source hash mismatch: {relative}')
        checked.append(relative)
    return {'method': method, 'status': 'PASS', 'revision': contract['revision'],
            'files_checked': checked, 'scope': 'Source identity only; not permission or experiment validation'}


def load_module(method, relative):
    """Import an explicitly allowlisted user-supplied module after validation.

    Calling this function may activate the upstream package's normal import behavior.
    The CLI validator never calls it. Use the method's recorded scientific environment.
    """
    validate(method)
    contract = contracts()[method]
    if relative not in contract['importable_files']:
        raise ValueError('Upstream module is not an approved integration entry point')
    directory = Path(os.environ['WMMD_UPSTREAM_ROOT']).resolve() / method
    name = '_wmmd_external_' + method + '_' + Path(relative).stem
    if name in sys.modules:
        return sys.modules[name]
    for item in contract.get('import_paths', ['.']):
        path = str(directory / item)
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, directory / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load validated upstream module')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_functions(method, relative, names, namespace):
    """Load allowlisted native pure functions without optional model/GAN imports.

    Function bodies are read from the validated official checkout at runtime;
    they are neither bundled nor rewritten. Dependencies are explicit in namespace.
    """
    validate(method)
    allowed = contracts()[method].get('function_exports', {}).get(relative, [])
    if not set(names).issubset(allowed):
        raise ValueError('Native function is not an approved integration entry point')
    file = Path(os.environ['WMMD_UPSTREAM_ROOT']) / method / relative
    tree = ast.parse(file.read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Pinned upstream function missing')
    scope = dict(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(file), 'exec'), scope)
    return {name: scope[name] for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=sorted(contracts()))
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.method, args.root)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'status': 'FAIL', 'method': args.method, 'error': str(error)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
