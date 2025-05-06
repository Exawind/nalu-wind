# Nalu-Wind 

[Documentation](https://exawind.github.io/nalu-wind/) | [Nightly test dashboard](http://my.cdash.org/index.php?project=Exawind) 

Nalu-Wind is a generalized, unstructured, massively parallel, incompressible
flow solver for wind turbine and wind farm simulations. The codebase is a
wind-focused fork of [NaluCFD](https://github.com/NaluCFD/Nalu); NaluCFD is developed 
and maintained by Sandia National Laboratories. Nalu-Wind is being actively
developed and maintained by a dedicated, multi-institutional team from [National
Renewable Energy Laboratory](https://nrel.gov), [Sandia National
Laboratories](https://sandia.gov), and [Univ. of Texas Austin](https://utexas.edu).

Nalu-Wind is developed as an open-source code with the following objectives: 

- an open, well-documented implementation of the state-of-the-art computational
  models for modeling wind farm flow physics at various fidelities that are
  backed by a comprehensive verification and validation (V&V) process;

- be capable of performing the highest-fidelity simulations of flowfields within
  wind farms; and 

- be able to leverage the high-performance leadership class computating
  facilities available at DOE national laboratories.

We hope that this community developed model will be used by research
laboratories, academia, and industry to develop the next-generation of wind farm
technologies. We welcome the wind energy community to use Nalu-Wind in their
research. When disseminating technical work that includes Nalu-Wind simulations
please reference the following citation:

    Sprague, M. A., Ananthan, S., Vijayakumar, G., Robinson, M., "ExaWind: A multifidelity 
    modeling and simulation environment for wind energy", NAWEA/WindTech 2019 Conference, 
    Amherst, MA, 2019.

## Part of the WETO Stack

Nalu-Wind is primarily developed with the support of the U.S. Department of Energy and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [High-Fidelity Modeling Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#high-fidelity-modeling)

## Documentation

Documentation is available online at https://exawind.github.io/nalu-wind/ and is
split into the following sections:

- [Theory manual](https://exawind.github.io/nalu-wind/source/theory/index.html):
  This section provides a detailed overview of the supported equation sets, the
  discretization and time-integration schemes, turbulence models available, etc.
  
- [Verification manual](https://exawind.github.io/nalu-wind/source/verification/index.html):
  This section documents the results from verification studies of the spatial
  and temporal schemes available in Nalu-Wind.
  
- [User manual](https://exawind.github.io/nalu-wind/source/user/index.html):
  The user manual contains detailed instructions on building the code, along
  with the required third-party libraries (TPLs) and usage.
  
All documentation is maintained alongside the source code within the git
repository and automatically deployed to a github-hosted website upon new commits.
  
## Compilation and usage

Nalu-Wind is primarily built upon the packages provided by the [Trilinos
project](https://trilinos.org), which in turn depends on several third-party
libraries (MPI, HDF5, NetCDF, parallel NetCDF), and YAML-CPP. In addition, it
has the following optional dependencies: hypre, TIOGA, and OpenFAST. Detailed
build instructions are available in the [user
manual](https://exawind.github.io/nalu-wind/source/user/building.html).
We recommend using [Spack](https://spack.io/) package manager to install
Nalu-Wind on your system.

### Testing and quality assurance

Nalu-Wind comes with a comprehensive unit test and regression test suite that
exercise almost all major components of the code. The `master` branch is
compiled and run through a regression test suite with different compilers
([GCC](https://gcc.gnu.org/), [LLVM/Clang](https://clang.llvm.org/), and
[Intel](https://software.intel.com/en-us/compilers)) on Linux and MacOS
operating systems, against both the `master` and `develop` branches of
[Trilinos](https://github.com/trilinos/Trilinos). Tests are performed both using
flat MPI and hybrid MPI-GPU hardware configurations. The results of the nightly
testing are publicly available on [CDash
dashboard](http://my.cdash.org/index.php?project=Exawind).

### Contributing, reporting bugs, and requesting help

To report issues or bugs please [create a new
issue](https://github.com/Exawind/nalu-wind/issues/new) on GitHub.

We welcome contributions from the community in form of bug fixes, feature
enhancements, documentation updates, etc. All contributions are processed
through pull-requests on GitHub. Please follow our [contributing
guidelines](https://github.com/Exawind/nalu-wind/blob/master/CONTRIBUTING.md)
when submitting pull-requests.
  
## License

Nalu-Wind is licensed under BSD 3-clause license. Please see the
[LICENSE](https://github.com/Exawind/nalu-wind/blob/master/LICENSE) included in
the source code repository for more details.

## Acknowledgements 

Nalu-Wind is currently being developed with funding from Department of Energy's
(DOE) Office of Science [Exascale Computing Project
(ECP)](https://www.exascaleproject.org/) and Energy Efficiency and Renewable
Energy (EERE) Wind Energy Technology Office (WETO). Please see [authors
file](https://github.com/Exawind/nalu-wind/blob/master/AUTHORS) for a 
list of contributors to Nalu-Wind. 
