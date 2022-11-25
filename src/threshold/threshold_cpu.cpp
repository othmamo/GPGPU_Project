
#include "../detect_obj.hpp"
#include "../threshold.hpp"
#include "../utils.hpp"

void apply_base_threshold(struct ImageMat* image, unsigned char threshold)
{
    for (int x = 0; x < image->height; ++x)
        for (int y = 0; y < image->width; ++y) {
            if (image->pixel[x][y] >= threshold)
                continue;
            else
                image->pixel[x][y] = 0;
        }
}

void apply_bin_threshold(struct ImageMat* in_image,
                         struct ImageMat* out_image,
                         unsigned char threshold)
{
    for (int x = 0; x < in_image->height; ++x)
        for (int y = 0; y < in_image->width; ++y) {
            out_image->pixel[x][y] = 255 * (in_image->pixel[x][y] >= threshold);
        }
}

void compute_otsu_threshold(struct ImageMat* in_image, struct ImageMat* temp)
{
    // TODO: Get two images for the connexe components
    // TODO: But that's after we do a full initial cleanup!

    unsigned char otsu_thresh = otsu_threshold(in_image);
    unsigned char otsu_thresh2 = otsu_thresh * 2.5;

    // First threshold saved to out_image_1
    apply_base_threshold(in_image, otsu_thresh - 10);

    // Second threshold saved to out_image_2
    apply_bin_threshold(in_image, temp, otsu_thresh2);
}

int compute_threshold(struct ImageMat* base_image, struct ImageMat* temp_image)
{
    compute_otsu_threshold(base_image, temp_image);
    int nb_compo = connexe_components(base_image, temp_image);

    swap_matrix(base_image, temp_image);

    return nb_compo;
}
