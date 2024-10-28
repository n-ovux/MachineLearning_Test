#include <math.h>
#include <stdio.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

void printMatrix(const gsl_matrix *matrix) {
  for (int i = 0; i < matrix->size1; i++) {
    for (int j = 0; j < matrix->size2; j++) {
      printf("%f, ", gsl_matrix_get(matrix, i, j));
    }
    printf("\n");
  }
}

void printVector(const gsl_vector *vector) {
  for (int i = 0; i < vector->size; i++) {
    printf("%f\n", gsl_vector_get(vector, i));
  }
}

gsl_matrix *createRandMatrix(int rows, int cols, int maxValue) {
  gsl_matrix *matrix = gsl_matrix_alloc(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      gsl_matrix_set(matrix, i, j, rand() % maxValue);
    }
  }
  return matrix;
}

gsl_vector *createRandVector(int rows, int maxValue) {
  gsl_vector *vector = gsl_vector_calloc(rows);
  for (int i = 0; i < rows; i++) {
    gsl_vector_set(vector, i, rand() % maxValue);
  }
  return vector;
}

static inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }

void applyFunctionMatrix(gsl_matrix *matrix, double (*function)(double x)) {
  for (int i = 0; i < matrix->size1; i++) {
    for (int j = 0; j < matrix->size2; j++) {
      gsl_matrix_set(matrix, i, j, function(gsl_matrix_get(matrix, i, j)));
    }
  }
}

void applyFunctionVector(gsl_vector *vector, double (*function)(double x)) {
  for (int i = 0; i < vector->size; i++) {
    gsl_vector_set(vector, i, function(gsl_vector_get(vector, i)));
  }
}

int main(void) {
  gsl_vector *input = createRandVector(3, 5);
  gsl_matrix *weights = createRandMatrix(3, 3, 5);
  gsl_vector *output = gsl_vector_calloc(3);

  gsl_blas_dgemv(CblasNoTrans, 1.0, weights, input, 0.0, output);
  printVector(output);
  applyFunctionVector(output, &sigmoid);
  printVector(output);

  gsl_vector_free(input);
  gsl_vector_free(output);
  gsl_matrix_free(weights);
  return 0;
}
