#pragma once
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void printMatrix(const gsl_matrix *matrix);
void printVector(const gsl_vector *vector);

gsl_matrix *createRandMatrix(int rows, int cols);
gsl_vector *createRandVector(int rows);

void elementWiseFunctionMatrix(gsl_matrix *matrix, double (*function)(double x));
void elementWiseFunctionVector(gsl_vector *vector, double (*function)(double x));
