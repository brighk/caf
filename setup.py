"""
Setup script for CAF installation.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causal-autonomy-framework",
    version="1.0.0",
    author="CAF Development Team",
    description="Sovereign Agent with Deterministic Output via Causal Grounding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/caf",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "torch>=2.1.2",
        "transformers>=4.37.0",
        "vllm>=0.2.7",
        "spacy>=3.7.2",
        "langchain>=0.1.4",
        "rdflib>=7.0.0",
        "SPARQLWrapper>=2.0.0",
        "chromadb>=0.4.22",
        "faiss-gpu>=1.7.2",
        "sentence-transformers>=2.3.1",
        "dowhy>=0.11.1",
        "networkx>=3.2.1",
        "prometheus-client>=0.19.0",
        "prometheus-fastapi-instrumentator>=6.1.0",
        "loguru>=0.7.2",
        "python-Levenshtein>=0.23.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "caf-api=api.main:app",
            "caf-inference=modules.inference_engine.server:app",
        ],
    },
)
