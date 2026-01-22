#include <vector>

// -------------- INPUT ROUTINES --------------
#include "math/matrix/matrix_ops.hpp"
#include "math/utilities.hpp"

// -------------- PLOTTING --------------
#include "io/plot/plot.hpp"
#include "utils/timing.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

double benchmark(int threads) {
    Eigen::setNbThreads(threads);
    Eigen::initParallel();

    std::cout << "Eigen reports threads = " << Eigen::nbThreads() << "\n";

    const int N = 2000;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd C(N, N);

    auto t1 = Clock::now();
    C.noalias() = A * B;
    auto t2 = Clock::now();

    std::chrono::duration<double> dt = t2 - t1;
    return dt.count();
}

int main() {
    std::cout << "=== Eigen Multithreading Test ===\n";
    std::cout << Eigen::SimdInstructionSetsInUse() << std::endl;
    double t1 = benchmark(1);
    std::cout << "Time with 1 thread: " << t1 << " seconds\n\n";

    double t8 = benchmark(8);
    std::cout << "Time with 8 threads: " << t8 << " seconds\n\n";

    if (t8 < t1 * 0.7)
        std::cout << ">>> Eigen IS using multithreading.\n";
    else
        std::cout << ">>> Eigen is NOT using multithreading.\n";

    return 0;
}
