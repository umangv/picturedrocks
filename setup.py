import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="picturedrocks",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Umang Varma, Anna Gilbert",
    author_email="uvarma@umich.edu",
    description="Single Cell RNA Sequencing Marker Selection Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/umangv/picturedrocks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "anndata",
        "colorlover",
        "h5py",
        "numpy",
        "pandas",
        "plotly",
        "scanpy",
        "scipy",
        "umap-learn"
    ],
    python_requires="~=3.6",
)
