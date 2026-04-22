# Kynema-UGF 

[Documentation](https://kynema.github.io/kynema-ugf/) | [Nightly test dashboard](http://my.cdash.org/index.php?project=Kynema) 

Kynema-UGF, wherein UGF stands for unstructured-grid fluid dynamics, is a generalized, unstructured-grid, massively parallel, incompressible-flow solver. The codebase was initiated in 2018 from [NaluCFD](https://github.com/NaluCFD/Nalu), which was developed by Sandia National Laboratories. Kynema-UGF is being actively
developed and maintained by a dedicated, multi-institutional team from [National Laboratory of the Rockies](https://nlr.gov) and [Sandia National Laboratories](https://sandia.gov).

Kynema-UGF was originally called Nalu-Wind (as part of the ExaWind stack), but was renamed in 2026 in order to reflect its broader, general computational-fluid-dynamics (CFD) capabilities. 

Kynema-UGF is developed as an open-source code with the following objectives: 

- an open, well-documented implementation of the state-of-the-art computational
  models for modeling flow physics relevant to energy systems that are
  backed by a comprehensive verification and validation (V&V) process;

- be able to leverage the high-performance leadership-class computing
  facilities available at DOE national laboratories.

When disseminating technical work that includes Kynema-UGF simulations
please reference the following citations:

The following contains the introduction of Nalu-Wind as part of the ExaWind suite:

    Sprague, M. A., Ananthan, S., Vijayakumar, G., Robinson, M., "ExaWind: A multifidelity 
    modeling and simulation environment for wind energy", NAWEA/WindTech 2019 Conference, 
    Amherst, MA, 2019. https://iopscience.iop.org/article/10.1088/1742-6596/1452/1/012071/pdf

The following contains details for Kynema-UGF (formerly Nalu-Wind):

    Sharma, A., M.J. Brazell, M.J., G. Vijayakumar, S. Ananthan, L. Cheung, N. deVelder, M.T. Henry de Frahan, N. Matula, P. Mullowney, J. Rood, P. Sakievich, A. Almgren, P.S. Crozier, and M.A. Sprague, 2024, ExaWind: Open-source CFD for hybrid-RANS/LES geometry-resolved wind turbine simulations in atmospheric flows. Wind Energy, 27, 225-257. https://onlinelibrary.wiley.com/doi/full/10.1002/we.2886.

## Documentation

Documentation is available online at https://kynema.github.io/kynema-ugf/ and is
split into the following sections:

- [Theory manual](https://kynema.github.io/kynema-ugf/source/theory/index.html):
  This section provides a detailed overview of the supported equation sets, the
  discretization and time-integration schemes, turbulence models available, etc.
  
- [Verification manual](https://kynema.github.io/kynema-ugf/source/verification/index.html):
  This section documents the results from verification studies of the spatial
  and temporal schemes available in Kynema-UGF.
  
- [User manual](https://kynema.github.io/kynema-ugf/source/user/index.html):
  The user manual contains detailed instructions on building the code, along
  with the required third-party libraries (TPLs) and usage.
  
All documentation is maintained alongside the source code within the git
repository and automatically deployed to a github-hosted website upon new commits.
  
## Compilation and usage

Kynema-UGF is primarily built upon the packages provided by the [Trilinos
project](https://trilinos.org), which in turn depends on several third-party
libraries (MPI, HDF5, NetCDF, parallel NetCDF), and YAML-CPP. In addition, it
has the following optional dependencies: hypre, TIOGA, and OpenFAST. Detailed
build instructions are available in the [user
manual](https://kynema.github.io/kynema-ugf/source/user/building.html).
We recommend using [Spack](https://spack.io/) package manager to install
Kynema-UGF on your system.

### Testing and quality assurance

Kynema-UGF comes with a comprehensive unit test and regression test suite that
exercise almost all major components of the code. The `master` branch is
compiled and run through a regression test suite with different compilers
([GCC](https://gcc.gnu.org/), [LLVM/Clang](https://clang.llvm.org/), and
[Intel](https://software.intel.com/en-us/compilers)) on Linux and MacOS
operating systems, against both the `master` and `develop` branches of
[Trilinos](https://github.com/trilinos/Trilinos). Tests are performed both using
flat MPI and hybrid MPI-GPU hardware configurations. The results of the nightly
testing are publicly available on [CDash
dashboard](http://my.cdash.org/index.php?project=Kynema).

### Contributing, reporting bugs, and requesting help

To report issues or bugs please [create a new
issue](https://github.com/kynema/kynema-ugf/issues/new) on GitHub.

We welcome contributions from the community in form of bug fixes, feature
enhancements, documentation updates, etc. All contributions are processed
through pull-requests on GitHub. Please follow our [contributing
guidelines](https://github.com/kynema/kynema-ugf/blob/master/CONTRIBUTING.md)
when submitting pull-requests.

To pass the formatting check, use this with a new version of `clang-format`:
```
find kynema-ugf.C unit_tests.C ./include ./src ./unit_tests \( -name "*.cpp" -o -name "*.H" -o -name "*.h" -o -name "*.C" \) -exec clang-format -i {} +
```
  
## License

Kynema-UGF is licensed under BSD 3-clause license. Please see the
[LICENSE](https://github.com/kynema/kynema-ugf/blob/master/LICENSE) included in
the source code repository for more details.

## Acknowledgements 

Kynema-UGF was originally developed with funding from Department of Energy's
(DOE) Office of Science [Exascale Computing Project
(ECP)](https://www.exascaleproject.org/) and Energy Efficiency and Renewable
Energy (EERE) Wind Energy Technology Office (WETO).  It is currently supported by the
DOE Office of Critical Minerals and Energy Innovation (CMEI). 

Please see [authors
file](https://github.com/kynema/kynema-ugf/blob/master/AUTHORS) for a 
list of contributors to Kynema-UGF, formerly Nalu-Wind.
