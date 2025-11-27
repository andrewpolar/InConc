#pragma once
#include <memory>
#include <vector>
#include <random>
#include <ctime>

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

struct Function {
	std::vector<double> f;
	double xmin;
	double xmax;
	double deltax;
	double offset;
	int index;
};

void InitializeFunction(Function& F, int nPoints,
	double xmin, double xmax, double fmin, double fmax, std::mt19937& rng) {

	F.f.resize(nPoints);
	std::uniform_real_distribution<double> dist(fmin, fmax);
	for (int j = 0; j < nPoints; ++j) {
		F.f[j] = dist(rng);
	}
	F.xmin = xmin;
	F.xmax = xmax;

	double gap = 0.01 * (F.xmax - F.xmin);
	F.xmin -= gap;
	F.xmax += gap;
	F.deltax = (F.xmax - F.xmin) / (nPoints - 1);
}

FORCE_INLINE double Compute(double x, bool freezeModel, Function& F) {
	if (!freezeModel) {
		bool isChanged = false;
		if (x <= F.xmin) {
			F.xmin = x;
			isChanged = true;
		}
		else if (x >= F.xmax) {
			F.xmax = x;
			isChanged = true;
		}
		if (isChanged) {
			double gap = 0.01 * (F.xmax - F.xmin);
			F.xmin -= gap;
			F.xmax += gap;
			F.deltax = (F.xmax - F.xmin) / (F.f.size() - 1);
		}
	}
	if (x <= F.xmin) {
		F.index = 0;
		F.offset = 0.001;
		return F.f[0];
	}
	else if (x >= F.xmax) {
		F.index = (int)(F.f.size()) - 2;
		F.offset = 0.999;
		return F.f[F.f.size() - 1];
	}
	else {
		double R = (x - F.xmin) / F.deltax;
		F.index = (int)(R);
		F.offset = R - F.index;
		return F.f[F.index] + (F.f[F.index + 1] - F.f[F.index]) * F.offset;
	}
}

FORCE_INLINE double ComputeDerivative(Function& F) {
	return (F.f[F.index + 1] - F.f[F.index]) / F.deltax;
}

FORCE_INLINE void Update(double delta, Function& F) {
	double tmp = delta * F.offset;
	F.f[F.index + 1] += tmp;
	F.f[F.index] += delta - tmp;
}

void CopyFunction(const std::unique_ptr<Function>& src, std::unique_ptr<Function>& dest) {
	for (size_t i = 0; i < src->f.size(); ++i) {
		dest->f[i] = src->f[i];
	}
	dest->xmin = src->xmin;
	dest->xmax = src->xmax;
	dest->deltax = src->deltax;
}

void CopyVector(const std::vector<std::unique_ptr<Function>>& src, std::vector<std::unique_ptr<Function>>& dest) {
	for (size_t i = 0; i < src.size(); ++i) {
		CopyFunction(src[i], dest[i]);
	}
}

std::unique_ptr<Function> CopyFunction(const std::unique_ptr<Function>& src) {
	return std::make_unique<Function>(*src);
}

std::vector<std::unique_ptr<Function>> CopyVector(const std::vector<std::unique_ptr<Function>>& src) {
	std::vector<std::unique_ptr<Function>> dst;
	dst.reserve(src.size());
	for (const auto& ptr : src) {
		dst.push_back(CopyFunction(ptr));
	}
	return dst;
}

void AddVectors(std::vector<std::unique_ptr<Function>>& target, const std::vector<std::unique_ptr<Function>>& source) {
	if (target.size() != source.size()) {
		throw std::runtime_error("Vector sizes do not match for averaging.");
	}
	for (size_t i = 0; i < target.size(); ++i) {
		Function& t = *target[i];
		const Function& s = *source[i];
		if (t.f.size() != s.f.size()) {
			throw std::runtime_error("Function vector sizes do not match.");
		}
		for (size_t j = 0; j < t.f.size(); ++j) {
			t.f[j] += s.f[j];
		}
		t.xmin += s.xmin;
		t.xmax += s.xmax;
		t.deltax += s.deltax;
	}
}

void ScaleVectors(std::vector<std::unique_ptr<Function>>& target, double scale) {
	for (size_t i = 0; i < target.size(); ++i) {
		Function& t = *target[i];
		for (size_t j = 0; j < t.f.size(); ++j) {
			t.f[j] *= scale;
		}
		t.xmin *= scale;
		t.xmax *= scale;
		t.deltax *= scale;
	}
}
