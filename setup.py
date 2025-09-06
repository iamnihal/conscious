from setuptools import setup, find_packages

setup(
    name="conscious",
    version="0.1.0",
    description="A syntax-aware code change analyzer with call graph generation",
    author="Innovation Week Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tree-sitter>=0.20.1,<0.21.0",
        "tree-sitter-languages>=1.8.0",
        "networkx>=3.1",
        "gitpython>=3.1.40",
        "psutil>=5.9.0",
        "click>=8.1.7",
        "rich>=13.7.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "conscious=conscious.cli:cli",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
