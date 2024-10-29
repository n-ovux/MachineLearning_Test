#include <math.h>
#include <stdio.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#include "util.h"

static inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }

int main(void) {
  int architecture[] = {2, 3, 1};
  int layerCount = sizeof(architecture) / sizeof(int);
  gsl_vector *layers[layerCount];
  gsl_matrix *weights[layerCount - 1];
  for (int i = 0; i < layerCount; i++) {
    layers[i] = gsl_vector_calloc(architecture[i]);
  }
  for (int i = 0; i < layerCount - 1; i++) {
    weights[i] = createRandMatrix(architecture[i + 1], architecture[i]);
  }

  gsl_vector *input = gsl_vector_calloc(2);
  gsl_vector_set(input, 0, 0.5);
  gsl_vector_set(input, 1, 0.5);
  gsl_vector_memcpy(layers[0], input);

  gsl_vector *output = gsl_vector_calloc(1);
  gsl_vector_set(output, 0, 0.25);

  for (int i = 0; i < layerCount - 1; i++) {
    gsl_blas_dgemv(CblasNoTrans, 1, weights[i], layers[i], 0, layers[i + 1]);
    elementWiseFunctionVector(layers[i + 1], &sigmoid);
  }

  double error = 0.0;
  for (int i = 0; i < architecture[layerCount - 1]; i++) {
    error += pow(gsl_vector_get(output, i) - gsl_vector_get(layers[layerCount - 1], i), 2);
  }
  error /= 2.0;

  printf("Input: \n");
  printVector(layers[0]);
  printf("Output: \n");
  printVector(layers[layerCount - 1]);
  printf("Error: %f\n", error);

  for (int i = 0; i < layerCount; i++) {
    gsl_vector_free(layers[i]);
  }
  for (int i = 0; i < layerCount - 1; i++) {
    gsl_matrix_free(weights[i]);
  }
  return 0;
}
