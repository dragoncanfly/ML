#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<mat.h>
#include"mkl.h"

typedef struct matrix
{
	float *data;
	int rows;
	int cols;
	int size;
	int byteSize;
}matrix;

enum hostError {
	hostErrNoError,
	hostErrError,
	hostErrFileOpenFailed,
	hostErrMemAlreadyAlloc
};
typedef hostError hostError_t;


hostError_t readFile(const char *path, const char *var, matrix *res);
hostError_t writeFile(float *data, int rows, int cols, const char *path, const char *var);
hostError_t writeFileInt(int *data, int rows, int cols, const char *path, const char *var);
void convert2Matrix(mxArray *ar, float *data);
void matrix_trans(float *data, float *data_trans, int m, int n);
void matrix_multi(float *pxp, float *p1, float *p2, int m, int n, int l);
void mean_col(float *p, float *p1, int m, int n);
void matrix_sum_sub(float *leftMatrix, float *rightMatrix, int m, int n, const char opera);
void repmat(float *ori, float *res, int ori_row, int ori_col, int m, int n);
void dot_div(float *p, float *p1, int m, int n, int k);

void pca_(float *X, int m, int n);
