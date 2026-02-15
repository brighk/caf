#!/usr/bin/env python3
"""
CAF Pre-Flight Check
====================
Comprehensive test suite to validate CAF is ready for deployment before
running expensive GPU experiments.

This test suite:
1. Checks all dependencies are installed
2. Tests SPARQL endpoint connectivity
3. Validates LLM can load on your hardware
4. Runs mini-experiment to verify full pipeline
5. Estimates resource requirements

Usage:
    # Quick check (no GPU)
    python tests/test_preflight_check.py

    # Full check with GPU test
    python tests/test_preflight_check.py --test-gpu

    # Check SPARQL only
    python tests/test_preflight_check.py --sparql-only

Requirements:
    - 4GB+ GPU for LLM tests
    - Running Fuseki instance for SPARQL tests
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_check(name: str, status: bool, details: str = ""):
    """Print check result."""
    symbol = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    status_text = f"{Colors.GREEN}PASS{Colors.END}" if status else f"{Colors.RED}FAIL{Colors.END}"
    print(f"{symbol} {name:.<50} {status_text}")
    if details:
        print(f"  {Colors.YELLOW}{details}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ WARNING: {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ ERROR: {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class PreFlightChecker:
    """Comprehensive pre-flight checker for CAF deployment."""

    def __init__(self, test_gpu: bool = False, test_sparql: bool = True):
        self.test_gpu = test_gpu
        self.test_sparql = test_sparql
        self.results = {}
        self.warnings = []
        self.errors = []

    def check_python_version(self) -> bool:
        """Check Python version is 3.12+."""
        version = sys.version_info
        required = (3, 12)

        is_ok = version >= required
        details = f"Found {version.major}.{version.minor}.{version.micro}"
        if not is_ok:
            details += f" (requires {required[0]}.{required[1]}+)"

        print_check("Python version >= 3.12", is_ok, details)
        return is_ok

    def check_dependencies(self) -> Dict[str, bool]:
        """Check all required Python packages are installed."""
        print_header("Checking Python Dependencies")

        required_packages = {
            # Core
            "fastapi": "FastAPI framework",
            "uvicorn": "ASGI server",
            "pydantic": "Data validation",

            # LLM
            "torch": "PyTorch",
            "transformers": "HuggingFace Transformers",
            "accelerate": "Model acceleration",
            "bitsandbytes": "Quantization",

            # SPARQL/RDF
            "rdflib": "RDF manipulation",
            "SPARQLWrapper": "SPARQL client",

            # NLP
            "spacy": "Entity extraction",

            # Vector DB
            "chromadb": "Vector database",

            # Causal
            "networkx": "Graph analysis",

            # Monitoring
            "prometheus_client": "Metrics",

            # Utilities
            "numpy": "Numerical computing",
            "pandas": "Data processing",
        }

        optional_packages = {
            "fuzzywuzzy": "Fuzzy string matching (optional)",
            "pytest": "Testing framework (dev)",
            "vllm": "Fast inference (optional)",
        }

        results = {}

        # Check required
        for package, description in required_packages.items():
            try:
                __import__(package.replace("-", "_"))
                print_check(f"{package:.<30} {description}", True)
                results[package] = True
            except ImportError:
                print_check(f"{package:.<30} {description}", False)
                results[package] = False
                self.errors.append(f"Missing required package: {package}")

        # Check optional
        print("\nOptional packages:")
        for package, description in optional_packages.items():
            try:
                __import__(package.replace("-", "_"))
                print_check(f"{package:.<30} {description}", True)
                results[package] = True
            except ImportError:
                print_warning(f"Optional package not installed: {package}")
                results[package] = False

        return results

    def check_spacy_model(self) -> bool:
        """Check spaCy model is downloaded."""
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
                print_check("spaCy model 'en_core_web_sm'", True, f"Version {spacy.__version__}")
                return True
            except OSError:
                print_check("spaCy model 'en_core_web_sm'", False,
                          "Run: python -m spacy download en_core_web_sm")
                self.errors.append("spaCy model not downloaded")
                return False
        except ImportError:
            print_check("spaCy model 'en_core_web_sm'", False, "spaCy not installed")
            return False

    def check_gpu_availability(self) -> Dict[str, any]:
        """Check GPU availability and CUDA setup."""
        print_header("Checking GPU/CUDA")

        gpu_info = {
            "available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "total_memory_gb": 0
        }

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            gpu_info["cuda_available"] = cuda_available

            print_check("CUDA available", cuda_available)

            if cuda_available:
                device_count = torch.cuda.device_count()
                gpu_info["device_count"] = device_count

                print_check(f"GPU devices found", device_count > 0, f"Count: {device_count}")

                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3

                    gpu_info["devices"].append({
                        "id": i,
                        "name": props.name,
                        "memory_gb": memory_gb,
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
                    gpu_info["total_memory_gb"] += memory_gb

                    print_check(f"GPU {i}: {props.name}", True,
                              f"{memory_gb:.1f}GB VRAM, Compute {props.major}.{props.minor}")

                gpu_info["available"] = True

                # Check if enough memory for 7B model with 4-bit quantization
                min_required = 4.0  # GB for 7B model with 4-bit
                if gpu_info["total_memory_gb"] < min_required:
                    print_warning(f"GPU memory ({gpu_info['total_memory_gb']:.1f}GB) may be insufficient for 7B model")
                    print_warning(f"Recommended: {min_required}GB+ for Llama-2-7b with 4-bit quantization")
                    self.warnings.append(f"Low GPU memory: {gpu_info['total_memory_gb']:.1f}GB")
                else:
                    print_info(f"GPU memory sufficient for 7B model with 4-bit quantization")
            else:
                print_error("No CUDA devices available")
                self.errors.append("No GPU available for LLM inference")

        except ImportError:
            print_check("PyTorch", False, "Not installed")
            self.errors.append("PyTorch not installed")

        return gpu_info

    def check_sparql_endpoint(self, endpoint: str = "http://localhost:3030") -> bool:
        """Check SPARQL endpoint is reachable."""
        print_header("Checking SPARQL Endpoint")

        try:
            import requests

            # Check if server is running
            try:
                response = requests.get(f"{endpoint}/$/ping", timeout=5)
                server_running = response.status_code == 200
                print_check(f"SPARQL server at {endpoint}", server_running)
            except requests.exceptions.ConnectionError:
                print_check(f"SPARQL server at {endpoint}", False, "Connection refused")
                self.warnings.append(f"SPARQL endpoint not reachable at {endpoint}")
                return False
            except Exception as e:
                print_check(f"SPARQL server at {endpoint}", False, str(e))
                return False

            # List datasets
            try:
                response = requests.get(f"{endpoint}/$/datasets", timeout=5)
                datasets = response.json()
                dataset_names = [d["ds.name"] for d in datasets.get("datasets", [])]

                print_check("SPARQL datasets available", len(dataset_names) > 0,
                          f"Found: {', '.join(dataset_names)}")

                # Check for conceptnet dataset
                if "/conceptnet" in dataset_names or "conceptnet" in dataset_names:
                    print_info("ConceptNet dataset found")
                else:
                    print_warning("ConceptNet dataset not found - you'll need to load data")
                    self.warnings.append("No ConceptNet dataset loaded")

            except Exception as e:
                print_warning(f"Could not list datasets: {e}")

            return server_running

        except ImportError:
            print_check("requests library", False, "Not installed")
            return False

    def test_sparql_query(self, endpoint: str = "http://localhost:3030/conceptnet/query") -> bool:
        """Test actual SPARQL query execution."""
        try:
            from experiments.real_fvl import RealFVL

            print("\nTesting SPARQL query execution...")

            fvl = RealFVL(sparql_endpoint=endpoint)

            # Test connection
            test_query = "ASK { ?s ?p ?o }"
            result = fvl._execute_sparql_query(test_query)

            if result.success:
                print_check("SPARQL query execution", True, f"Latency: {result.latency_ms:.1f}ms")

                # Get triple count
                count_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
                count_result = fvl._execute_sparql_query(count_query)
                if count_result.success:
                    count = count_result.result.get("results", {}).get("bindings", [{}])[0].get("count", {}).get("value", "0")
                    print_info(f"Dataset contains {count} triples")

                    if int(count) == 0:
                        print_warning("Dataset is empty - load ConceptNet data first")
                        self.warnings.append("SPARQL dataset is empty")

                return True
            else:
                print_check("SPARQL query execution", False, result.error)
                return False

        except ImportError as e:
            print_check("Real FVL import", False, str(e))
            return False
        except Exception as e:
            print_check("SPARQL query execution", False, str(e))
            return False

    def test_llm_loading(self, model_size: str = "7b") -> bool:
        """Test if LLM can be loaded on available hardware."""
        print_header("Testing LLM Loading (this may take a few minutes)")

        if not self.test_gpu:
            print_info("Skipping GPU test (use --test-gpu to enable)")
            return True

        try:
            from experiments.llm_integration import create_llama_layer

            print_info(f"Attempting to load Llama-2-{model_size}-chat with 4-bit quantization...")
            print_info("This will download the model if not cached (~3.5GB for 7b)")

            start_time = time.time()

            try:
                llm = create_llama_layer(
                    model_size=model_size,
                    use_4bit=True,
                    open_source=True
                )

                load_time = time.time() - start_time
                print_check(f"LLM loading (Llama-2-{model_size})", True,
                          f"Loaded in {load_time:.1f}s")

                # Test inference
                print_info("Testing inference...")
                test_prompt = "What is 2+2?"

                try:
                    response = llm.generate(test_prompt)
                    print_check("LLM inference", True, f"Response: {response[:50]}...")
                    return True
                except Exception as e:
                    print_check("LLM inference", False, str(e))
                    self.errors.append(f"LLM inference failed: {e}")
                    return False

            except Exception as e:
                print_check(f"LLM loading (Llama-2-{model_size})", False, str(e))
                self.errors.append(f"Failed to load LLM: {e}")
                return False

        except ImportError as e:
            print_check("LLM integration", False, f"Import error: {e}")
            return False

    def test_mini_experiment(self) -> bool:
        """Run minimal experiment with 2 chains to test full pipeline."""
        print_header("Running Mini Experiment (2 chains)")

        try:
            from experiments.run_experiment import ExperimentRunner
            from experiments.caf_algorithm import CAFLoop, CAFConfig

            print_info("Creating experiment runner...")

            runner = ExperimentRunner(
                output_dir="tests/preflight_results",
                seed=42,
                verbose=False
            )

            print_info("Generating 2 synthetic chains...")
            runner.generate_dataset(num_chains=2, perturbations_per_chain=1)

            print_check("Dataset generation", True, "2 chains created")

            print_info("Running CAF verification (simulation mode)...")
            start_time = time.time()

            caf_outputs, baseline_outputs = runner.run_verification(use_baselines=False)

            duration = time.time() - start_time

            print_check("CAF verification", True, f"Completed in {duration:.1f}s")

            # Check outputs
            if len(caf_outputs) == 2:
                print_check("Output validation", True, "All chains processed")

                # Compute metrics
                print_info("Computing metrics...")
                metrics = runner.compute_metrics()

                print_info(f"Sample metrics:")
                print_info(f"  - Entailment accuracy: {metrics['CAF'].entailment_accuracy:.2%}")
                print_info(f"  - Contradiction rate: {metrics['CAF'].contradiction_rate:.2%}")

                return True
            else:
                print_check("Output validation", False, f"Expected 2 outputs, got {len(caf_outputs)}")
                return False

        except Exception as e:
            print_check("Mini experiment", False, str(e))
            self.errors.append(f"Mini experiment failed: {e}")
            import traceback
            print_error(traceback.format_exc())
            return False

    def estimate_full_experiment_resources(self, num_chains: int = 75) -> Dict:
        """Estimate resources needed for full experiment."""
        print_header("Resource Estimation for Full Experiment")

        # Based on mini experiment, extrapolate
        estimates = {
            "num_chains": num_chains,
            "estimated_time_minutes": 0,
            "estimated_gpu_memory_gb": 0,
            "estimated_disk_space_mb": 0
        }

        # Simulation mode (no GPU)
        if not self.test_gpu:
            # ~1-2 seconds per chain in simulation
            estimates["estimated_time_minutes"] = (num_chains * 1.5) / 60
            estimates["estimated_gpu_memory_gb"] = 0
            estimates["estimated_disk_space_mb"] = 50

            print_info(f"Simulation mode (no GPU):")
            print_info(f"  - Estimated time: {estimates['estimated_time_minutes']:.1f} minutes")
            print_info(f"  - GPU memory: Not required")
            print_info(f"  - Disk space: ~{estimates['estimated_disk_space_mb']}MB")

        # Real LLM mode
        else:
            # ~30-60 seconds per chain with 7B model
            estimates["estimated_time_minutes"] = (num_chains * 45) / 60
            estimates["estimated_gpu_memory_gb"] = 4.5  # 7B model with 4-bit
            estimates["estimated_disk_space_mb"] = 100

            print_info(f"Real LLM mode (Llama-2-7b, 4-bit):")
            print_info(f"  - Estimated time: {estimates['estimated_time_minutes']:.1f} minutes ({estimates['estimated_time_minutes']/60:.1f} hours)")
            print_info(f"  - GPU memory needed: ~{estimates['estimated_gpu_memory_gb']:.1f}GB")
            print_info(f"  - Disk space: ~{estimates['estimated_disk_space_mb']}MB")

        return estimates

    def generate_report(self) -> Dict:
        """Generate comprehensive pre-flight report."""
        print_header("Pre-Flight Check Summary")

        total_errors = len(self.errors)
        total_warnings = len(self.warnings)

        if total_errors == 0 and total_warnings == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED{Colors.END}")
            print(f"{Colors.GREEN}System is ready for full experiment deployment!{Colors.END}")
            status = "READY"
        elif total_errors == 0:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ CHECKS PASSED WITH WARNINGS{Colors.END}")
            print(f"{Colors.YELLOW}System can run but some features may be limited{Colors.END}")
            status = "READY_WITH_WARNINGS"
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ CHECKS FAILED{Colors.END}")
            print(f"{Colors.RED}System is NOT ready - fix errors before deployment{Colors.END}")
            status = "NOT_READY"

        print(f"\nErrors: {total_errors}")
        for error in self.errors:
            print(f"  {Colors.RED}✗ {error}{Colors.END}")

        print(f"\nWarnings: {total_warnings}")
        for warning in self.warnings:
            print(f"  {Colors.YELLOW}⚠ {warning}{Colors.END}")

        report = {
            "status": status,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save report
        report_path = Path("tests/preflight_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{Colors.BLUE}Report saved to: {report_path}{Colors.END}")

        return report

    def run_all_checks(self) -> bool:
        """Run all pre-flight checks."""
        print_header("CAF Pre-Flight Check")
        print_info(f"Testing mode: {'GPU' if self.test_gpu else 'Simulation only'}")
        print_info(f"SPARQL: {'Enabled' if self.test_sparql else 'Disabled'}")

        all_passed = True

        # 1. Python version
        if not self.check_python_version():
            all_passed = False

        # 2. Dependencies
        deps = self.check_dependencies()
        if not all(deps.values()):
            all_passed = False

        # 3. spaCy model
        if not self.check_spacy_model():
            all_passed = False

        # 4. GPU (if testing)
        gpu_info = self.check_gpu_availability()
        if self.test_gpu and not gpu_info["available"]:
            all_passed = False

        # 5. SPARQL (if testing)
        if self.test_sparql:
            sparql_ok = self.check_sparql_endpoint()
            if sparql_ok:
                self.test_sparql_query()

        # 6. LLM loading (if testing GPU)
        if self.test_gpu:
            if not self.test_llm_loading():
                all_passed = False

        # 7. Mini experiment
        if not self.test_mini_experiment():
            all_passed = False

        # 8. Resource estimation
        self.estimate_full_experiment_resources()

        # 9. Generate report
        self.generate_report()

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="CAF Pre-Flight Check - Validate system before deployment"
    )
    parser.add_argument(
        "--test-gpu",
        action="store_true",
        help="Test GPU and LLM loading (requires 4GB+ GPU)"
    )
    parser.add_argument(
        "--sparql-only",
        action="store_true",
        help="Only test SPARQL endpoint"
    )
    parser.add_argument(
        "--no-sparql",
        action="store_true",
        help="Skip SPARQL tests"
    )
    parser.add_argument(
        "--model-size",
        default="7b",
        choices=["7b", "8b", "13b"],
        help="LLM model size to test (if --test-gpu)"
    )

    args = parser.parse_args()

    if args.sparql_only:
        # Quick SPARQL check
        checker = PreFlightChecker(test_gpu=False, test_sparql=True)
        checker.check_sparql_endpoint()
        checker.test_sparql_query()
        return

    # Full check
    checker = PreFlightChecker(
        test_gpu=args.test_gpu,
        test_sparql=not args.no_sparql
    )

    success = checker.run_all_checks()

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
