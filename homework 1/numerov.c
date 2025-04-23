#include <math.h>
#include <stddef.h>

// Function prototype (declare V before use)
static inline double V(double r, double s);

// Constants
static const double epsilon = 5.99;
static const double a = 2.1041471061733663;
static const double b = 2.1236506034448093;

// Lennard-Jones potential
static inline double V(double r, double s) {
    const double sr = s / r;
    const double sr6 = pow(sr, 6);
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

// Numerov solver (rest of the code remains the same)
void numerov(double E, double h, int l, const double* r, const double* s, size_t n, double* y)  {
    double k[n];
    const double h_sq_12 = h * h / 12.0;
    const double sigma = s[0];  // Assume s is constant (σ in LJ potential)

    // Precompute k[j] = (1/a)(E - V(r,s)) - l(l+1)/r²
    for (size_t j = 2; j < n; j++) {
        k[j] = (1.0 / a) * (E - V(r[j], sigma)) - l * (l + 1) / (r[j] * r[j]);
    }

    // Initial condition: y[j] = exp(-(b/r[j])⁵) for r[j] ≤ σ/2
    for (size_t j = 0; j < n; j++) {
        if (r[j] <= sigma / 2.0) {
            y[j] = exp(-pow(b / r[j], 5));
        } else {
            y[j] = 0.0;  // Explicit initialization
        }
    }

    // Numerov iteration (avoids branching in inner loop)
    for (size_t j = 1; j < n - 1; j++) {
        if (r[j + 1] > sigma / 2.0) {
            const double denom = 1.0 + h_sq_12 * k[j + 1];
            y[j + 1] = (
                -y[j - 1] * (1.0 + h_sq_12 * k[j - 1]) + 
                2.0 * y[j] * (1.0 - 5.0 * h_sq_12 * k[j])
            ) / denom;
        }
    }
}