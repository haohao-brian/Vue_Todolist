#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <cstring>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, comm_sz = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_sz);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (rank == 0 && argc != 4) {
        std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        MPI_Abort(comm, 1);
    }

    std::string input_img, output_img, output_txt;
    if (rank == 0) {
        input_img = argv[1];
        output_img = argv[2];
        output_txt = argv[3];
    }

    auto start = std::chrono::high_resolution_clock::now();

    // --- root loads image ---
    Image img;
    int width = 0, height = 0, channels = 0;
    if (rank == 0) {
        img = Image(input_img);
        width = img.width;
        height = img.height;
        channels = img.channels;
    }

    // broadcast shape/meta
    MPI_Bcast(&width,    1, MPI_INT, 0, comm);
    MPI_Bcast(&height,   1, MPI_INT, 0, comm);
    MPI_Bcast(&channels, 1, MPI_INT, 0, comm);

    if (width == 0 || height == 0) {
        if (rank == 0) std::cerr << "Invalid image dimensions.\n";
        MPI_Finalize();
        return 1;
    }

    const int N = width * height;      // number of pixels (not elements)
    // We’ll compute grayscale: 1 float per pixel.
    // If input has 3 channels, we’ll scatter each plane separately.

    // build counts/displs for pixel-slices (uneven split ok)
    std::vector<int> counts(comm_sz), displs(comm_sz);
    {
        int base = N / comm_sz, rem = N % comm_sz, off = 0;
        for (int r = 0; r < comm_sz; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
    }
    const int nloc = counts[rank];

    // local buffers
    std::vector<float> local_gray(nloc);
    std::vector<float> local_r, local_g, local_b;

    // root-only global gray buffer for gather
    std::vector<float> gray_data; // size N
    if (rank == 0) gray_data.resize(N);

    // If the source image is already single-channel, we can just scatter that one plane.
    if (channels == 1) {
        // Scatter one plane (img.data has N floats)
        MPI_Scatterv(
            (rank == 0 ? img.data : nullptr), counts.data(), displs.data(), MPI_FLOAT,
            local_gray.data(), nloc, MPI_FLOAT,
            0, comm
        );
    } else {
        // Assume planar layout as your original indexing implies:
        // R plane [0..N-1], G plane [N..2N-1], B plane [2N..3N-1]
        // If your Image is interleaved, convert to planar on root first.

        local_r.resize(nloc);
        local_g.resize(nloc);
        local_b.resize(nloc);

        // scatter each plane
        MPI_Scatterv(
            (rank == 0 ? img.data + 0 * N : nullptr), counts.data(), displs.data(), MPI_FLOAT,
            local_r.data(), nloc, MPI_FLOAT, 0, comm);
        MPI_Scatterv(
            (rank == 0 ? img.data + 1 * N : nullptr), counts.data(), displs.data(), MPI_FLOAT,
            local_g.data(), nloc, MPI_FLOAT, 0, comm);
        MPI_Scatterv(
            (rank == 0 ? img.data + 2 * N : nullptr), counts.data(), displs.data(), MPI_FLOAT,
            local_b.data(), nloc, MPI_FLOAT, 0, comm);

        // local grayscale
        for (int i = 0; i < nloc; ++i) {
            local_gray[i] = 0.299f * local_r[i] + 0.587f * local_g[i] + 0.114f * local_b[i];
        }
    }

    // gather grayscale back to root
    MPI_Gatherv(
        local_gray.data(), nloc, MPI_FLOAT,
        (rank == 0 ? gray_data.data() : nullptr), counts.data(), displs.data(), MPI_FLOAT,
        0, comm
    );

    // --- after gather, only root needs the full image for SIFT ---
    std::vector<Keypoint> kps;
    Image result;
    if (rank == 0) {
        Image gray(width, height, 1);
        // attach/copy the gathered data into gray
        // If Image takes ownership, you could move; otherwise copy:
        std::memcpy(gray.data, gray_data.data(), sizeof(float) * N);

        // run SIFT on grayscale image
        kps = find_keypoints_and_descriptors(gray);

        // validation output
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
        }

        result = draw_keypoints(gray, kps);
        result.save(output_img);
    }

    MPI_Finalize();

    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }
    return 0;
}
