import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="invert",
    version="0.0.1",
    author="Kiril Bykov",
    author_email="kirill079@gmail.com",
    description="Labeling Neural Representations with Inverse Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lapalap/invert",
    packages=setuptools.find_packages(),
    install_requires=required,
    python_requires=">=3.6",
    include_package_data=True,
    keywords=["machine learning", "neural networks", "representation analysis"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
)
