import os

from setuptools import setup, find_packages

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "src", "version.py"),
    encoding="utf-8",
) as f:
    exec(f.read())

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as r:
    README = r.read()

test_deps = ["pytest", "pytest-httpserver", "werkzeug", "statsig-python-core"]
extras = {
    "test": test_deps,
}

setup(
    name="statsig_ai",
    # pylint: disable=undefined-variable
    version=__version__,  # type: ignore
    description="Statsig Python Server SDK",
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
        "statsig-python-core",
        "opentelemetry-api>=1.27.0",
        "opentelemetry-sdk>=1.27.0",
        "opentelemetry-exporter-otlp-proto-http>=1.27.0",
        "opentelemetry-semantic-conventions>=0.48b0",
    ],
    tests_require=test_deps,
    extras_require=extras,
    include_package_data=True,
    packages=find_packages(include=["statsig_ai"]),
    python_requires=">=3.9",
)
