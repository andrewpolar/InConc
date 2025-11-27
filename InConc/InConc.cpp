//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

//This is a demo of new paradigm of Kolmogorov-Arnold networks - infinite concurrency. 
//This is only a demo of concept, this code is not designed for lightning fast execution. 
//The idea is to start as many threads as hardware allows and train them on short batches 
//of random records or disjoints. Once any two are finished, the models averaged and 
//threads restarted. While training, all models tend to a fixed point.
//Even this primitive demo is significantly faster anything other. 
//The features are 4 by 4 random matrices, targets are determinants, training set 100K,
//validation set 20K.

#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include "Helper.h"
#include "Function.h"

double g_pearson = 0.0;

void ValidationDeterminant(const std::vector<std::unique_ptr<Function>>& inner,
    const std::vector<std::unique_ptr<Function>>& outer, const std::vector<std::vector<double>>& features,
    const std::vector<double>& targets, int nInner, int nOuter) {

    size_t nRecords = targets.size();
    size_t nFeatures = features[0].size();
    std::vector<double> models0(nInner);
    std::vector<double> models1(nOuter);
    std::vector<double> predictions(nRecords);

    for (size_t record = 0; record < nRecords; ++record) {
        for (int k = 0; k < nInner; ++k) {
            models0[k] = 0.0;
            for (size_t j = 0; j < nFeatures; ++j) {
                models0[k] += Compute(features[record][j], true, *inner[k * nFeatures + j]);
            }
            models0[k] /= nFeatures;
        }
        for (int k = 0; k < nOuter; ++k) {
            models1[k] = 0.0;
            for (int j = 0; j < nInner; ++j) {
                models1[k] += Compute(models0[j], true, *outer[j]);
            }
            models1[k] /= nInner;
        }
        predictions[record] = models1[0];
    }
    g_pearson = Pearson(predictions, targets);
}

void TrainingDeterminant(std::vector<std::unique_ptr<Function>>& inner,
    std::vector<std::unique_ptr<Function>>& outer, const std::vector<std::vector<double>>& features,
    const std::vector<double>& targets, int nInner, int nOuter, int start, int end, int nRecords, double alpha, double accuracy) {

    size_t nFeatures = features[0].size();
    std::vector<double> models0(nInner);
    std::vector<double> models1(nOuter);
    std::vector<double> deltas0(nInner);
    std::vector<double> deltas1(nOuter);

    for (int idx = start; idx < end; ++idx) {
        int record = idx;
        if (record >= nRecords) record -= nRecords;
        for (int k = 0; k < nInner; ++k) {
            models0[k] = 0.0;
            for (size_t j = 0; j < nFeatures; ++j) {
                models0[k] += Compute(features[record][j], false, *inner[k * nFeatures + j]);
            }
            models0[k] /= nFeatures;
        }
        for (int k = 0; k < nOuter; ++k) {
            models1[k] = 0.0;
            for (int j = 0; j < nInner; ++j) {
                models1[k] += Compute(models0[j], false, *outer[j]);
            }
            models1[k] /= nInner;
        }
        deltas1[0] = alpha * (targets[record] - models1[0]);
        if (std::abs(deltas1[0]) < accuracy) continue;
        for (int j = 0; j < nInner; ++j) {
            deltas0[j] = deltas1[0] * ComputeDerivative(*outer[j]);
        }
        for (int k = 0; k < nOuter; ++k) {
            for (int j = 0; j < nInner; ++j) {
                Update(deltas1[k], *outer[j]);
            }
        }
        for (int k = 0; k < nInner; ++k) {
            for (size_t j = 0; j < nFeatures; ++j) {
                Update(deltas0[k], *inner[k * nFeatures + j]);
            }
        }
    }
}

std::vector<std::pair<int, int>> make_random_pairs(int N) {
    std::vector<int> v(N);
    for (int i = 0; i < N; ++i) v[i] = i;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(v.begin(), v.end(), gen);

    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(N / 2);

    for (int i = 0; i < N; i += 2) {
        pairs.emplace_back(v[i], v[i + 1]);
    }
    return pairs;
}

void DeterminantsParallel() {
    //1.dataset
    const int nTrainingRecords = 100'000;
    const int nValidationRecords = 20'000;
    const int nMatrixSize = 4;
    const int nFeatures = nMatrixSize * nMatrixSize;
    const double min = 0.0;
    const double max = 10.0;

    //2.network
    const int nInner = 70;
    const int nOuter = 1;
    const double alpha = 0.2;
    const int nInnerPoints = 3;
    const int nOuterPoints = 25;
    const double termination = 0.97;

    //3.batches. all constants are arbitrary
    const int nBatchSize = 5'000;
    const int nBatches = 16;
    const int nLoops = 80;
    /////////////////////

    //data generation
    printf("Generating data ...\n");
    auto features_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto features_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
    auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize);
    auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize);
    printf("Data is ready ...\n");

    //processing start
    using Clock = std::chrono::steady_clock;
    auto start_application = Clock::now();

    double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
    double targetMax = *std::max_element(targets_training.begin(), targets_training.end());
    double accuracy = std::abs(targetMin);
    if (accuracy < std::abs(targetMax)) accuracy = std::abs(targetMax);
    accuracy *= 0.01;

    std::random_device rd;
    std::mt19937 rng(rd());

    //create containers sized to nBatches
    std::vector<std::vector<std::unique_ptr<Function>>> inners;
    std::vector<std::vector<std::unique_ptr<Function>>> outers;

    inners.resize(1);   // make sure index 0 exists
    outers.resize(1);

    //generate one set as random
    inners[0].reserve(nInner * nFeatures);
    for (int i = 0; i < nInner * nFeatures; ++i) {
        auto function = std::make_unique<Function>();
        InitializeFunction(*function, nInnerPoints, min, max, targetMin, targetMax, rng);
        inners[0].push_back(std::move(function));
    }

    outers[0].reserve(nInner);
    for (int i = 0; i < nInner; ++i) {
        auto function = std::make_unique<Function>();
        InitializeFunction(*function, nOuterPoints, targetMin, targetMax, targetMin, targetMax, rng);
        outers[0].push_back(std::move(function));
    }

    //copy to remaining sets
    for (int b = 1; b < nBatches; ++b) {
        inners.push_back(CopyVector(inners[0]));
        outers.push_back(CopyVector(outers[0]));
    }

    printf("Parallel version\n");
    printf("Targets are determinants of random %d * %d matrices, %d training records\n",
        nMatrixSize, nMatrixSize, nTrainingRecords);
    int start = 0;
    std::vector<std::thread> threads;
    for (int loop = 0; loop < nLoops; ++loop) {
        // concurrent training of model copies
        threads.clear();
        for (int b = 0; b < nBatches; ++b) {
            int threadStart = start;
            int threadEnd = start + nBatchSize;
            // Launch thread to train inners[b] and outers[b]
            threads.emplace_back(TrainingDeterminant, std::ref(inners[b]), std::ref(outers[b]),
                std::cref(features_training), std::cref(targets_training),
                nInner, nOuter, threadStart, threadEnd, nTrainingRecords, alpha, accuracy);

            // advance start for next batch (wrap-around)
            start += nBatchSize;
            if (start >= nTrainingRecords) start -= nTrainingRecords;
        }

        for (auto& t : threads) {
            t.join();
        }

        auto pairs = make_random_pairs(nBatches);

        for (auto& p : pairs) {
            AddVectors(inners[p.first], inners[p.second]);
            AddVectors(outers[p.first], outers[p.second]);
            ScaleVectors(inners[p.first], 1.0 / 2.0);
            ScaleVectors(outers[p.first], 1.0 / 2.0);
            CopyVector(inners[p.first], inners[p.second]);
            CopyVector(outers[p.first], outers[p.second]);
        }

        auto current = Clock::now();
        double elapsed = std::chrono::duration<double>(current - start_application).count();
        printf("Loop = %d,  pearson = %4.3f, time = %2.3f\n", loop, g_pearson, elapsed);
        if (g_pearson >= termination) break;
    }
    printf("Validation ...\n");
    ValidationDeterminant(inners[0], outers[0], features_validation, targets_validation, nInner, nOuter);
    printf("Pearson %f\n\n", g_pearson);
}

int main() {
    DeterminantsParallel();
}
