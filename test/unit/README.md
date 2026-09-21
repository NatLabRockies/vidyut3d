# Unit tests

Tests of the header-only helpers in `Source/` (`HelperFuncs.H`,
`compute_explicit_flux.H`) and of the mechanism interface (`PlasmaChem`,
`Chemistry.H`). There is no dependency other than AMReX; `UnitTest.H` is a small
harness with `UT_TEST`, `UT_CHECK` and `UT_CHECK_CLOSE`.

## Build and run

```
make -j 4
./unittests3d.gnu.ex
```

The executable prints one line per test and returns the number of failed tests,
so a non-zero exit code means failure. `DIM=1` and `DIM=2` also work.

Run a subset with a name filter:

```
./unittests3d.gnu.ex unittest.filter=weno
```

## Mechanism

`Chemistry.H`, `ProbParm.H` and `UserFunctions.H` are taken from an existing
case folder, `test/verification/He_RF_1d` by default. To run the mechanism
tests (species lookup, charge conservation of the reaction set, signs of the
transport coefficients) against another case:

```
make realclean
make -j 4 CASE_DIR=../models/Ar_DBD
```

## Adding a test

Add a `UT_TEST(name) { ... }` block to `main.cpp`. Functions that are
`AMREX_GPU_DEVICE` only have to be evaluated through `unittest::device_eval`,
which runs them in an `amrex::ParallelFor` and copies the results to the host.
