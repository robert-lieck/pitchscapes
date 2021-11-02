import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
   install_requires = fh.read().splitlines()

setuptools.setup(
    name="pitchscapes",
    version="0.1.7",
    author="Robert Lieck",
    author_email="robert.lieck@epfl.ch",
    description="computing pitch-scapes and pitch-scape clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robert-lieck/pitchscapes",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

