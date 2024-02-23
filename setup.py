from glob import glob
from setuptools import setup

from pybind11.setup_helpers import intree_extensions
from pybind11.setup_helpers import build_ext

ext_modules = intree_extensions(glob('multinomial_cobweb/*.cpp'))

# Specify the C++ standard for each extension module
# for module in ext_modules:
#     module.cxx_std = '2a'
#     module.extra_link_args.append("-ltbb")

setup(
    name="multinomial_cobweb",
    author="Xin Lian, Christopher J. MacLellan",
    author_email="xinthelian@hotmail.com, maclellan.christopher@gmail.com",
    url="https://github.com/xinthelian/cobweb-psych",
    description="The Cobweb version used for psychological effect experiments in human categorization learning",
    long_description=open('README.md').read(),
    description_content_type="text/x-rst; charset=UTF-8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
    keywords="clustering,machine-learning,human-learning,cognitive-science",
    license="MIT",
    license_file="LICENSE.txt",
    packages=["multinomial_cobweb"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["pybind11", "numpy", "pandas", "scipy"],
)
