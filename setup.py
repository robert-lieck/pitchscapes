import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pitchscapes",
    version="0.1.3",
    author="Robert Lieck",
    author_email="robert.lieck@epfl.ch",
    description="computing pitch-scapes and pitch-scape clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DCMLab/pitchscapes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

