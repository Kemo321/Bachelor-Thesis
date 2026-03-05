#include <benchmark/benchmark.h>
#include <numeric>
#include <vector>

static void BM_SimpleVectorAdd(benchmark::State& state)
{
    std::vector<float> a(state.range(0), 1.0f);
    std::vector<float> b(state.range(0), 2.0f);
    std::vector<float> res(state.range(0));

    for (auto _ : state)
    {
        for (size_t i = 0; i < a.size(); ++i)
        {
            res[i] = a[i] + b[i];
        }
        benchmark::DoNotOptimize(res.data());
    }

    state.SetBytesProcessed(long(state.iterations()) * long(state.range(0)) * sizeof(float));
}

BENCHMARK(BM_SimpleVectorAdd)->Range(1024, 1024 * 1024);

BENCHMARK_MAIN();
