from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)


setup(
    name="smallssd",
    description="Open source agricultural data",
    author="Gabriel Tseng",
    author_email="gabrieltseng95@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smallrobotcompany/smallssd",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    packages=["smallssd"],
    install_requires=[
        # https://github.com/pytorch/pytorch/issues/78362
        "protobuf==3.20.1",
        "tqdm>=4.61.1",
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "pillow>=9.1.0",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
