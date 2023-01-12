Preable (subst)
=======
# syn

[![Release](https://img.shields.io/github/v/release/wraith1995/syntax)](https://img.shields.io/github/v/release/wraith1995/syntax)
[![Build status](https://github.com/wraith1995/syntax/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/wraith1995/syntax/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/wraith1995/syntax/branch/main/graph/badge.svg)](https://codecov.io/gh/wraith1995/syntax)
[![Commit activity](https://img.shields.io/github/commit-activity/m/wraith1995/syntax)](https://img.shields.io/github/commit-activity/m/wraith1995/syntax)
[![License](https://img.shields.io/github/license/wraith1995/syntax)](https://img.shields.io/github/license/wraith1995/syntax)

Thing.... Description

- **Github repository**: <https://github.com/wraith1995/syntax/>
- **Documentation** <https://wraith1995.github.io/syntax/>

## Getting started with your project

First, create a repository on GitHub with the same name as this project, and then run the following commands:

``` bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:wraith1995/syn.git
git push -u origin main
```

Finally, install the environment and the pre-commit hooks with 

```bash
make install
```

You are now ready to start development on your project! The CI/CD
pipeline will be triggered when you open a pull request, merge to main,
or when you create a new release.

To finalize the set-up for publishing to PyPi or Artifactory, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting 
[this page](https://github.com/wraith1995/syn/settings/secrets/actions/new).
- Create a [new release](https://github.com/wraith1995/syn/releases/new) on Github. 


# TODO (in no particular order)
0. Resolve serious disagreements on the nature of validation beyond types. The possibility of internal type conversion, types of types and their handling, shittyness of python typing module.
1. Different types of iteration methods: depth vs. breadth, external or internal, dup vs no dup, dup notion, ocality, local ordering, Nones, Names, Flattening. General Collections.abc interfance questions (sets, functions, etc...)
2. The problem of matching over an IR that does not yet exist.
3. Extension of existing ADTs or merging of ADTS
4. Basic documentation.
5. Vistior patterns
6. Custom show or at least better show. 
7. Python front end instead of text. See (2).
8. Functorial IRs
9. Logging intergration
10. Github integrations
11. Variations on the lambda calc implementation for examples. Add these to tests.
12. Attrs vs. Dataclasses. Attrs might save some boiler plate, but it outside standard python.
13. Greater variation on internal errors and consistenty with python.
14. psf black
15. Ref Mutability and selective mutalability/frozeness - ability to express cfg
16. Integration with mypy/pyright/whwatever
17. Function recursion helpers
18. Folding/tree systems


Pretty printing:
Expr, stmt, graph
Create a new tag in the form ``*.*.*``.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).

