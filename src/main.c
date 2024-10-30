#include <math.h>
#include <stdio.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#include "util.h"

static inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }

gsl_vector *forward(const int architecture[], int layerCount, gsl_vector *layers[], gsl_matrix *weights[]) {
  for (int i = 0; i < layerCount - 1; i++) {
    gsl_blas_dgemv(CblasNoTrans, 1, weights[i], layers[i], 0, layers[i + 1]);
    elementWiseFunctionVector(layers[i + 1], &sigmoid);
  }
  gsl_vector *output = gsl_vector_calloc(layers[layerCount - 1]->size);
  gsl_vector_memcpy(output, layers[layerCount - 1]);
  return output;
}
double error(const gsl_vector *output, const gsl_vector *expectedOutput) {
  double error = 0.0;
  for (int i = 0; i < output->size; i++) {
    error += pow(gsl_vector_get(expectedOutput, i) - gsl_vector_get(output, i), 2);
  }
  error /= 2.0;
  return error;
}

int main(void) {
  int architecture[] = {2, 3, 1};
  int layerCount = sizeof(architecture) / sizeof(int);
  gsl_vector *layers[layerCount];
  gsl_matrix *weights[layerCount - 1];
  gsl_matrix *weightsCopy[layerCount - 1];
  gsl_matrix *weightsPrime[layerCount - 1];
  for (int i = 0; i < layerCount; i++) {
    layers[i] = gsl_vector_calloc(architecture[i]);
  }
  for (int i = 0; i < layerCount - 1; i++) {
    weights[i] = createRandMatrix(architecture[i + 1], architecture[i]);
    weightsPrime[i] = gsl_matrix_calloc(architecture[i + 1], architecture[i]);
    weightsCopy[i] = gsl_matrix_calloc(architecture[i + 1], architecture[i]);
  }

  gsl_vector *input = gsl_vector_calloc(2);
  gsl_vector_set(input, 0, 0.5);
  gsl_vector_set(input, 1, 0.5);
  gsl_vector_memcpy(layers[0], input);
  gsl_vector *expectedOutput = gsl_vector_calloc(1);
  gsl_vector_set(expectedOutput, 0, 0.25);

  double h = 0.001;
  double learningRate = 1;
  int epochs = 100;
  for (int epoch = 0; epoch < epochs; epoch++) {
    gsl_vector *output = forward(architecture, layerCount, layers, weights);
    for (int layer = 0; layer < layerCount - 1; layer++) {
      for (int row = 0; row < weightsPrime[layer]->size1; row++) {
        for (int col = 0; col < weightsPrime[layer]->size2; col++) {
          for (int i = 0; i < layerCount - 1; i++) {
            gsl_matrix_memcpy(weightsCopy[i], weights[i]);
          }
          gsl_matrix_set(weightsCopy[layer], row, col, gsl_matrix_get(weightsCopy[layer], row, col) + h);
          gsl_vector *outputShifted = forward(architecture, layerCount, layers, weightsCopy);
          double dE = error(outputShifted, expectedOutput) - error(output, expectedOutput);
          gsl_matrix_set(weightsPrime[layer], row, col, dE / h);
        }
      }
    }
    for (int layer = 0; layer < layerCount - 1; layer++) {
      for (int row = 0; row < weights[layer]->size1; row++) {
        for (int col = 0; col < weights[layer]->size2; col++) {
          gsl_matrix_set(weights[layer], row, col,
                         gsl_matrix_get(weights[layer], row, col) - learningRate * gsl_matrix_get(weightsPrime[layer], row, col));
        }
      }
    }
  }

  gsl_vector *output = forward(architecture, layerCount, layers, weights);
  double outputError = error(output, expectedOutput);
  printf("Input: \n");
  printVector(layers[0]);
  printf("Output: \n");
  printVector(layers[layerCount - 1]);
  printf("Error: %f\n", outputError);

  for (int i = 0; i < layerCount; i++) {
    gsl_vector_free(layers[i]);
  }
  for (int i = 0; i < layerCount - 1; i++) {
    gsl_matrix_free(weights[i]);
  }
  return 0;
}
