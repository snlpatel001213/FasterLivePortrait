from setuptools import setup, find_packages

setup(
    name="kokoro",
    version="0.9.4",
    description="TTS",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="hexgrad",
    author_email="hello@hexgrad.com",
    url="https://github.com/hexgrad/kokoro",
    project_urls={
        "Homepage": "https://github.com/hexgrad/kokoro",
        "Repository": "https://github.com/hexgrad/kokoro",
    },
    license_files=["LICENSE"],
    packages=find_packages(include=["kokoro", "kokoro.*"]),
    python_requires=">=3.10, <3.13",
    install_requires=[
        "huggingface_hub",
        "loguru",
        "misaki[en]>=0.9.4",
        "numpy",
        "torch",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "kokoro = kokoro.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
