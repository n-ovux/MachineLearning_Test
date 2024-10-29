#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "util.h"

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

gsl_matrix *createRandMatrix(int rows, int cols) {
  gsl_matrix *matrix = gsl_matrix_alloc(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      gsl_matrix_set(matrix, i, j, (rand() % 100) / 100.0);
    }
  }
  return matrix;
}

gsl_vector *createRandVector(int rows) {
  gsl_vector *vector = gsl_vector_calloc(rows);
  for (int i = 0; i < rows; i++) {
    gsl_vector_set(vector, i, (rand() % 100) / 100.0);
  }
  return vector;
}

void elementWiseFunctionMatrix(gsl_matrix *matrix, double (*function)(double x)) {
  for (int i = 0; i < matrix->size1; i++) {
    for (int j = 0; j < matrix->size2; j++) {
      gsl_matrix_set(matrix, i, j, function(gsl_matrix_get(matrix, i, j)));
    }
  }
}

void elementWiseFunctionVector(gsl_vector *vector, double (*function)(double x)) {
  for (int i = 0; i < vector->size; i++) {
    gsl_vector_set(vector, i, function(gsl_vector_get(vector, i)));
  }
}
