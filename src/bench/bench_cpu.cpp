#include "detect_obj.hpp"
#include "helpers_images.hpp"

#include <benchmark/benchmark.h>
#include <iostream>

std::string ref_image_path = "../images/base.png";
int width, height, channels;

unsigned char* ref_image = load_image(const_cast<char*>(ref_image_path.c_str()),
                                      &width,
                                      &height,
                                      &channels);

void BM_gray_scale_cpu(benchmark::State& st)
{
    struct ImageMat* image = new_matrix(height, width);

    for (auto _ : st) {
        to_gray_scale(ref_image, image, width, height, 3);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_blurring_cpu(benchmark::State& st)
{
    struct ImageMat* image = new_matrix(height, width);
    struct ImageMat* temp_image = new_matrix(height, width);

    struct GaussianKernel* g_kernel = create_gaussian_kernel(5);

    for (auto _ : st)
        apply_blurring(image, temp_image, g_kernel);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_difference_cpu(benchmark::State& st)
{
    struct ImageMat* image1 = new_matrix(height, width);
    struct ImageMat* image2 = new_matrix(height, width);

    for (auto _ : st)
        difference(image1, image2);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_closing_cpu(benchmark::State& st)
{
    struct MorphologicalKernel* k1 = circular_kernel(5);
    struct ImageMat* image1 = new_matrix(height, width);
    struct ImageMat* image2 = new_matrix(height, width);

    for (auto _ : st) {
        perform_erosion(image1, image2, k1);
        perform_dilation(image1, image2, k1);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_opening_cpu(benchmark::State& st)
{
    struct MorphologicalKernel* k1 = circular_kernel(11);
    struct ImageMat* image1 = new_matrix(height, width);
    struct ImageMat* image2 = new_matrix(height, width);

    for (auto _ : st) {
        perform_dilation(image1, image2, k1);
        perform_erosion(image1, image2, k1);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_threshold_cpu(benchmark::State& st)
{
    struct ImageMat* image1 = new_matrix(height, width);
    struct ImageMat* image2 = new_matrix(height, width);

    for (auto _ : st) {
        threshold(image1, image2);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_main_cpu(benchmark::State& st)
{
    std::string ref_image_path = "../images/base.png";
    std::string obj_image_path = "../images/obj.png";

    int width, height, channels;
    unsigned char* ref_image = load_image(
        const_cast<char*>(ref_image_path.c_str()), &width, &height, &channels);
    unsigned char* obj_image = load_image(
        const_cast<char*>(obj_image_path.c_str()), &width, &height, &channels);

    unsigned char** images =
        (unsigned char**)std::malloc(sizeof(unsigned char*) * 2);
    images[0] = ref_image;
    images[1] = obj_image;

    int* nb_objs = (int*)std::malloc(1 * sizeof(int));

    for (auto _ : st)
        main_detection(images, 2, width, height, channels, nb_objs);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gray_scale_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_blurring_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_difference_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_closing_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_opening_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_threshold_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_main_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
