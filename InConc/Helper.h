#pragma once
#include <memory>
#include <vector>
#include <cfloat>
#include <random>
#include <ctime>

double Pearson(const std::vector<double>& x, const std::vector<double>& y) {
    int len = (int)x.size();
    double xmean = 0.0;
    double ymean = 0.0;
    for (int i = 0; i < len; ++i) {
        xmean += x[i];
        ymean += y[i];
    }
    xmean /= len;
    ymean /= len;

    double covariance = 0.0;
    for (int i = 0; i < len; ++i) {
        covariance += (x[i] - xmean) * (y[i] - ymean);
    }

    double stdX = 0.0;
    double stdY = 0.0;
    for (int i = 0; i < len; ++i) {
        stdX += (x[i] - xmean) * (x[i] - xmean);
        stdY += (y[i] - ymean) * (y[i] - ymean);
    }
    stdX = sqrt(stdX);
    stdY = sqrt(stdY);
    return covariance / stdX / stdY;
}

void ShowMatrix(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%5.3f ", matrix[i][j]);
        }
        printf("\n");
    }
}
void ShowVector(const std::vector<double>& ptr) {
    size_t N = ptr.size();
    int cnt = 0;
    for (size_t i = 0; i < N; ++i) {
        printf("%5.2f ", ptr[i]);
        if (++cnt >= 10) {
            printf("\n");
            cnt = 0;
        }
    }
}

//determinants
static std::vector<std::vector<double>> GenerateInput(int nRecords, int nFeatures, double min, double max) {
    constexpr int nThreads = 8; // fixed inside function
    auto x = std::vector<std::vector<double>>(nRecords);

    // allocate rows first
    for (int i = 0; i < nRecords; ++i) {
        x[i] = std::vector<double>(nFeatures);
    }

    // Prepare different seeds (generated on the parent thread)
    std::random_device rd;
    std::vector<std::uint64_t> seeds(nThreads);
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int t = 0; t < nThreads; ++t) {
        // mix rd() with time and thread index to avoid identical seeds
        seeds[t] = (static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^ (now + (std::uint64_t)t * 0x9e3779b97f4a7c15ULL);
    }

    // launch anonymous worker threads (capturing seed by value)
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        threads.emplace_back(
            [t, nThreads, nRecords, nFeatures, min, max, &x, seed = seeds[t]]()
            {
                std::mt19937_64 rng(seed);
                std::uniform_real_distribution<double> dist(min, max);
                for (int i = t; i < nRecords; i += nThreads) {
                    for (int j = 0; j < nFeatures; ++j) {
                        x[i][j] = dist(rng);
                    }
                }
            }
        );
    }

    for (auto& th : threads) th.join();
    return x;
}

static double determinant(const std::vector<std::vector<double>>& matrix) {
    size_t n = matrix.size();
    if (n == 1) {
        return matrix[0][0];
    }
    if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    double det = 0.0;
    for (size_t col = 0; col < n; ++col) {
        std::vector<std::vector<double>> subMatrix(n - 1, std::vector<double>(n - 1));
        for (size_t i = 1; i < n; ++i) {
            int subCol = 0;
            for (size_t j = 0; j < n; ++j) {
                if (j == col) continue;
                subMatrix[i - 1][subCol++] = matrix[i][j];
            }
        }
        det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * determinant(subMatrix);
    }
    return det;
}

static double ComputeDeterminant(const std::vector<double>& input, int N) {
    std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = input[cnt++];
        }
    }
    return determinant(matrix);
}

static std::vector<double> ComputeDeterminantTarget(
    const std::vector<std::vector<double>>& x,
    int nMatrixSize)
{
    size_t nRecords = x.size();
    constexpr int nThreads = 8;  // fixed inside
    auto target = std::vector<double>(nRecords);

    // Worker: each thread handles every nThreads-th record
    auto worker = [&](int tid) {
        for (size_t i = tid; i < nRecords; i += nThreads) {
            target[i] = ComputeDeterminant(x[i], nMatrixSize);
        }
        };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        threads.emplace_back(worker, t);
    }

    for (auto& th : threads) th.join();

    return target;
}
//end determinants

///////// Areas of faces of tetrahedron
double Area(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3) {
    double a1 = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
    double a2 = (x2 - x1) * (z3 - z1) - (z2 - z1) * (x3 - x1);
    double a3 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    double A = 0.5 * sqrt(a1 * a1 + a2 * a2 + a3 * a3);
    return A;
}
std::vector<std::vector<double>> MakeRandomMatrix(int rows, int cols, double min, double max) {
    std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<std::vector<double>> matrix(rows);
    for (int i = 0; i < rows; ++i) {
        matrix[i] = std::vector<double>(cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(rng);
        }
    }
    return matrix;
}
std::vector<std::vector<double>> ComputeTargetMatrix(const std::vector<std::vector<double>>& X) {
    size_t rows = X.size();
    std::vector<std::vector<double>> matrix(rows);
    for (size_t i = 0; i < rows; ++i) {
        matrix[i] = std::vector<double>(4);
        matrix[i][0] = Area(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], X[i][6], X[i][7], X[i][8]);
        matrix[i][1] = Area(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], X[i][9], X[i][10], X[i][11]);
        matrix[i][2] = Area(X[i][0], X[i][1], X[i][2], X[i][6], X[i][7], X[i][8], X[i][9], X[i][10], X[i][11]);
        matrix[i][3] = Area(X[i][3], X[i][4], X[i][5], X[i][6], X[i][7], X[i][8], X[i][9], X[i][10], X[i][11]);
    }
    return matrix;
}
//////////// End tetrahedron
