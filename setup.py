import os

from setuptools import setup, find_packages

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "statsig_ai", "version.py"),
    encoding="utf-8",
) as f:
    exec(f.read())

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as r:
    README = r.read()

test_deps = [
    "pytest",
    "pytest-httpserver",
    "werkzeug",
    "statsig-python-core",
    "pytest-asyncio",
    "openai",
]
extras = {
    "test": test_deps,
}

setup(
    name="statsig_ai",
    # pylint: disable=undefined-variable
    version=__version__,  # type: ignore
    description="Statsig AI Python SDK",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="https://github.com/statsig-io/statsig-ai-python",
    license="ISC",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "requests",
        "statsig-python-core",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
        "typing-extensions>=4.1.0",
    ],
    tests_require=test_deps,
    extras_require=extras,
    include_package_data=True,
    packages=find_packages(include=["statsig_ai", "statsig_ai.*"]),
    python_requires=">=3.9",
)
