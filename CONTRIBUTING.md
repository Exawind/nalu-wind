# Contributing to Nalu-Wind

Thank you for considering contributing to Nalu-Wind project. Please follow these
guidelines for contributing to the Nalu-Wind project.

## Reporting bugs

This section guides you through the process of [submitting a bug
report](https://github.com/exawind/nalu-wind/issues/new) for the Nalu-Wind project.
Following these guidelines will help maintainers understand your issue,
reproduce the behavior, and develop a fix in an expedient fashion. Before
submitting your bug report, please perform a [cursory
search](https://github.com/search?q=is%3Aissue+repo%3Aexawind%2Fnalu-wind) to
see if the problem has been already reported. If it has been reported, and the
issue is still open, add a comment to the existing issue instead of opening a
new issue.

### Tips for effective bug reporting

- Use a clear descriptive title for the issue

- Describe the steps to reproduce the problem, the behavior you observed after
  following the steps, and the expected behavior

- Provide the SHA ID of the git commit Nalu-Wind code that you are using, as
  well as the SHA ID of the Trilinos build

- Provide as much detail as possible about the operating system, compiler
  versions, and third-party libraries used to build Nalu-Wind
  
- Include output of the CMake configuration step 

- For build errors, include the complete output when executing `make`

- For runtime errors, provide a stack trace of the error output

## Contributing code and documentation changes

Contributions can take the form of bug fixes, feature enhancements,
documentation updates. All updates to the repository are managed via [pull
requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).
One of the easiest ways to get started is by looking at [open
issues](https://github.com/Exawind/nalu-wind/issues) and contributing fixes,
enhancements that address those issues. If your code contribution involves large
changes or additions to the codebase, we recommend opening an issue first and
discussing your proposed changes with the core development team to ensure that
your efforts are well directed, and so that your submission can be reviewed and
merged seamlessly by the maintenance team.

Nalu-Wind can be developed locally. For instructions please consult the
[developer
manual](https://nalu-wind.readthedocs.io/en/latest/source/developer/index.html).

### Guidelines for preparing and submitting pull-requests

- Use a clear descriptive title for your pull-requests

- Describe if your submission is a bugfix, documentation update, or a feature
  enhancement. Provide a concise description of your proposed changes. 
  
- Provide references to open issues, if applicable, to provide the necessary
  context to understand your pull request
  
- Make sure that your pull-request merges cleanly with the `master` branch of
  Nalu-Wind. When working on a feature, always create your feature branch off of
  the latest `master` commit
  
- Ensure that the code compiles without warnings, the unit tests and regression
  tests all pass without errors, and the documentation builds properly with your
  modifications
  
- New physics models and code enhancements should be accompanied with relevant
  updates to the documentation, supported by necessary verification and
  validation, as well as unit tests and regression tests
  
- Where appropriate please use [Clang
  format](https://clang.llvm.org/docs/ClangFormat.html) to format your code to
  match the rest of Nalu-Wind. Only do that for sections you edit/add. Don't run
  `clang-format` on the entire file if it is not a new file created by you.
  
Once a pull-request is submitted you will iterate with Nalu-Wind maintainers
until your changes are in an acceptable state and can be merged in. You can push
addditional commits to the branch used to create the pull-request to reflect the
feedback from maintainers and users of the code.
  
