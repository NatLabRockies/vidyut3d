#ifndef Vidyut_unittest_H_
#define Vidyut_unittest_H_

#include <string>
#include <functional>
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <AMReX_Gpu.H>

// Minimal test harness, no dependency other than AMReX.
//
//   UT_TEST(name) { ... UT_CHECK(cond); UT_CHECK_CLOSE(a, b, tol); ... }
//
// registers a test. The executable returns the number of failed tests.
namespace unittest {

struct TestCase
{
    std::string name;
    std::function<void()> func;
};

inline amrex::Vector<TestCase>& registry()
{
    static amrex::Vector<TestCase> tests;
    return tests;
}

inline int& nfailed_checks()
{
    static int nfail = 0;
    return nfail;
}

struct Registrar
{
    Registrar(const std::string& name, std::function<void()> func)
    {
        registry().push_back({name, func});
    }
};

inline void
report_failure(const std::string& what, const char* file, const int line)
{
    nfailed_checks()++;
    amrex::Print() << "    FAILED: " << what << " (" << file << ":" << line
                   << ")\n";
}

inline void check_close(
    const amrex::Real val,
    const amrex::Real ref,
    const amrex::Real tol,
    const char* expr,
    const char* file,
    const int line)
{
    // relative for large reference values, absolute near zero
    const amrex::Real scale = amrex::max(amrex::Real(1.0), std::abs(ref));
    if (!(std::abs(val - ref) <= tol * scale))
    {
        report_failure(
            std::string(expr) + ": got " + std::to_string(val) + ", expected " +
                std::to_string(ref),
            file, line);
    }
}

// Evaluate func(n, out) for n in [0, nvals) on the device (a plain loop in a
// CPU build) and return the values on the host. Use this for functions that
// are AMREX_GPU_DEVICE only. func has to be device callable, i.e. a lambda
// declared as [=] AMREX_GPU_DEVICE(int n, amrex::Real* out).
template <typename F>
amrex::Vector<amrex::Real> device_eval(const int nvals, F const& func)
{
    amrex::Gpu::DeviceVector<amrex::Real> dvals(nvals, 0.0);
    amrex::Real* dptr = dvals.data();
    amrex::ParallelFor(
        nvals, [=] AMREX_GPU_DEVICE(int n) noexcept { func(n, dptr); });
    amrex::Gpu::streamSynchronize();
    amrex::Vector<amrex::Real> hvals(nvals);
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, dvals.begin(), dvals.end(), hvals.begin());
    return hvals;
}

inline int run_all(const std::string& filter)
{
    int ntests = 0;
    int nfailed_tests = 0;
    for (auto const& test : registry())
    {
        if (!filter.empty() && test.name.find(filter) == std::string::npos)
        {
            continue;
        }
        ntests++;
        const int nfail_before = nfailed_checks();
        amrex::Print() << "[ RUN  ] " << test.name << "\n";
        test.func();
        if (nfailed_checks() > nfail_before)
        {
            nfailed_tests++;
            amrex::Print() << "[ FAIL ] " << test.name << "\n";
        } else
        {
            amrex::Print() << "[  OK  ] " << test.name << "\n";
        }
    }
    amrex::Print() << "\n"
                   << ntests - nfailed_tests << " of " << ntests
                   << " tests passed\n";
    if (ntests == 0)
    {
        // a mistyped filter must not look like a successful run
        amrex::Print() << "FAILED: no test matches unittest.filter = \""
                       << filter << "\"\n";
        return 1;
    }
    return nfailed_tests;
}
} // namespace unittest

#define UT_TEST(name)                                                          \
    void ut_func_##name();                                                     \
    static unittest::Registrar ut_reg_##name(#name, ut_func_##name);           \
    void ut_func_##name()

#define UT_CHECK(cond)                                                         \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            unittest::report_failure(#cond, __FILE__, __LINE__);               \
        }                                                                      \
    } while (false)

#define UT_CHECK_CLOSE(val, ref, tol)                                          \
    unittest::check_close(val, ref, tol, #val, __FILE__, __LINE__)

#endif
