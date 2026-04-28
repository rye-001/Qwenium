#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import argparse
import re

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_command(cmd, cwd=None, timeout=300):
    """Runs a command and returns (returncode, stdout, stderr)"""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=cwd,
            text=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return -1, stdout, stderr + "\nTIMEOUT EXPIRED"
    except Exception as e:
        return -2, "", str(e)

def find_test_binaries(build_dir):
    bin_dir = os.path.join(build_dir, "bin")
    if not os.path.exists(bin_dir):
        return []
    
    binaries = []
    for f in os.listdir(bin_dir):
        path = os.path.join(bin_dir, f)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            if f == "qwenium":
                continue  # CLI executable, not a test
            if f.endswith("-tests") or f.endswith("-test"):
                binaries.append((f, path))
    return sorted(binaries)

def main():
    parser = argparse.ArgumentParser(description="Run Qwen-3 Inference Engine Tests")
    parser.add_argument("--build-dir", default="build", help="Path to build directory")
    parser.add_argument("--filter", help="Filter tests by name (regex)")
    parser.add_argument("--verbose", action="store_true", help="Show full test output")
    parser.add_argument("--build", action="store_true", help="Run build.sh before testing")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, args.build_dir)

    print(f"{Colors.BOLD}Running lint checks...{Colors.ENDC}")
    rc, out, err = run_command("./scripts/check_recipe_isolation.sh", cwd=project_root)
    if rc != 0:
        print(f"{Colors.FAIL}Lint failed: check_recipe_isolation{Colors.ENDC}")
        print(out)
        sys.exit(1)
    print(f"{Colors.OKGREEN}  check_recipe_isolation: PASSED{Colors.ENDC}")

    if args.build:
        print(f"{Colors.BOLD}Building project...{Colors.ENDC}")
        rc, out, err = run_command("./scripts/build.sh", cwd=project_root)
        if rc != 0:
            print(f"{Colors.FAIL}Build failed!{Colors.ENDC}")
            print(err)
            sys.exit(1)

    binaries = find_test_binaries(build_dir)
    if not binaries:
        print(f"{Colors.WARNING}No test binaries found in {build_dir}/bin{Colors.ENDC}")
        print("Run with --build to compile them first.")
        sys.exit(1)

    if args.filter:
        regex = re.compile(args.filter)
        binaries = [b for b in binaries if regex.search(b[0])]

    print(f"{Colors.HEADER}{Colors.BOLD}Running {len(binaries)} Test Binaries...{Colors.ENDC}\n")

    results = []
    total_start = time.time()

    for name, path in binaries:
        print(f"[{Colors.OKBLUE}RUNNING{Colors.ENDC}] {name}...", end="", flush=True)
        start = time.time()
        
        # Determine if it's GTest or custom
        # (We just run them and check return code for now)
        rc, out, err = run_command(path)
        duration = time.time() - start
        
        if rc == 0:
            print(f"\r[{Colors.OKGREEN}  PASS  {Colors.ENDC}] {name} ({duration:.2f}s)")
            results.append((name, True, duration))
        else:
            print(f"\r[{Colors.FAIL}  FAIL  {Colors.ENDC}] {name} ({duration:.2f}s) - Exit Code: {rc}")
            results.append((name, False, duration))
            if not args.verbose:
                # Print a snippet of the error if not verbose
                print(f"{Colors.WARNING}Output snippet:{Colors.ENDC}")
                lines = out.splitlines() + err.splitlines()
                for line in lines[-10:]:
                    print(f"  {line}")
        
        if args.verbose:
            print("-" * 40)
            if out: print(out)
            if err: print(err)
            print("-" * 40)

    total_duration = time.time() - total_start
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    passed = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    for name, success, duration in results:
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if success else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"  {status} {name:<40} {duration:.2f}s")
    
    print(f"\n{Colors.BOLD}Total: {len(results)} | {Colors.OKGREEN}Passed: {len(passed)}{Colors.ENDC} | {Colors.FAIL}Failed: {len(failed)}{Colors.ENDC}")
    print(f"Total time: {total_duration:.2f}s")

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
