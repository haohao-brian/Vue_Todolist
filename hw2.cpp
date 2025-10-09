#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "image.hpp"
#include "sift.hpp"

struct PackedKeypoint {
    int i;
    int j;
    int octave;
    int scale;
    float x;
    float y;
    float sigma;
    float extremum_val;
    uint8_t descriptor[128];
};

static PackedKeypoint pack_keypoint(const Keypoint& kp)
{
    PackedKeypoint packed{};
    packed.i = kp.i;
    packed.j = kp.j;
    packed.octave = kp.octave;
    packed.scale = kp.scale;
    packed.x = kp.x;
    packed.y = kp.y;
    packed.sigma = kp.sigma;
    packed.extremum_val = kp.extremum_val;
    std::copy(kp.descriptor.begin(), kp.descriptor.end(), packed.descriptor);
    return packed;
}

static Keypoint unpack_keypoint(const PackedKeypoint& packed)
{
    Keypoint kp{};
    kp.i = packed.i;
    kp.j = packed.j;
    kp.octave = packed.octave;
    kp.scale = packed.scale;
    kp.x = packed.x;
    kp.y = packed.y;
    kp.sigma = packed.sigma;
    kp.extremum_val = packed.extremum_val;
    std::copy(std::begin(packed.descriptor), std::end(packed.descriptor), kp.descriptor.begin());
    return kp;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc != 4) {
        if (mpi_rank == 0) {
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];

    Image img(input_img);
    img = img.channels == 1 ? img : rgb_to_grayscale(img);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Keypoint> local_kps = find_keypoints_and_descriptors(
        img, SIGMA_MIN, N_OCT, N_SPO, C_DOG, C_EDGE, LAMBDA_ORI, LAMBDA_DESC, mpi_rank, mpi_size);

    std::vector<PackedKeypoint> packed_local;
    packed_local.reserve(local_kps.size());
    for (const auto& kp : local_kps) {
        packed_local.push_back(pack_keypoint(kp));
    }

    int local_bytes = static_cast<int>(packed_local.size() * sizeof(PackedKeypoint));
    std::vector<int> recv_counts;
    if (mpi_rank == 0) {
        recv_counts.resize(mpi_size, 0);
    }

    MPI_Gather(&local_bytes, 1, MPI_INT,
               mpi_rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs;
    int total_bytes = 0;
    if (mpi_rank == 0) {
        displs.resize(mpi_size, 0);
        for (int idx = 0; idx < mpi_size; ++idx) {
            displs[idx] = total_bytes;
            total_bytes += recv_counts[idx];
        }
    }

    std::vector<PackedKeypoint> packed_global;
    if (mpi_rank == 0 && total_bytes > 0) {
        int total_keypoints = total_bytes / static_cast<int>(sizeof(PackedKeypoint));
        packed_global.resize(total_keypoints);
    }

    MPI_Gatherv(packed_local.empty() ? nullptr : packed_local.data(), local_bytes, MPI_BYTE,
                (mpi_rank == 0 && !packed_global.empty()) ? packed_global.data() : nullptr,
                mpi_rank == 0 ? recv_counts.data() : nullptr,
                mpi_rank == 0 ? displs.data() : nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<Keypoint> kps;
    if (mpi_rank == 0 && !packed_global.empty()) {
        kps.reserve(packed_global.size());
        for (const auto& pkp : packed_global) {
            kps.push_back(unpack_keypoint(pkp));
        }
        std::sort(kps.begin(), kps.end(), [](const Keypoint& a, const Keypoint& b) {
            return std::tie(a.octave, a.scale, a.i, a.j)
                 < std::tie(b.octave, b.scale, b.i, b.j);
        });
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    /////////////////////////////////////////////////////////////
    // The following code is for the validation
    // You can not change the logic of the following code, because it is used for judge system
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
        ofs.close();
    }

    Image result = draw_keypoints(img, kps);
    result.save(output_img);
    /////////////////////////////////////////////////////////////

    std::chrono::duration<double, std::milli> duration = end - start;
    if (mpi_rank == 0) {
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }

    MPI_Finalize();
    return 0;
}
