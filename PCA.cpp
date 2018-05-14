#include"pca.h"

hostError_t readFile(const char *path, const char *var, matrix *result)
{
	MATFile *fmat = NULL;
	fmat = matOpen(path, "r");
	if (fmat == NULL)
	{
		return hostErrFileOpenFailed;
	}
	mxArray *ar;
	ar = matGetVariable(fmat, var);
	if (ar == NULL) {
		return hostErrError;
	}
	else {

		int rows = mxGetM(ar);
		int cols = mxGetN(ar);
		result->data = (float*)malloc(sizeof(float)*rows*cols);
		convert2Matrix(ar, result->data);
		result->rows = rows;
		result->cols = cols;
		result->size = rows*cols;
		result->byteSize = result->size * sizeof(float);

		mxDestroyArray(ar);
		return hostErrNoError;
	}
}

hostError_t writeFile(float* data, int rows, int cols, const char *path, const char *var)
{
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL) return hostErrFileOpenFailed;
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j*rows + i] = (float)data[i*cols + j];
		}
	}
	int status = matPutVariable(pw, var, br);
	if (status != 0)
	{
		//printf("%s", "save result failed");
		return hostErrError;
	}
	mxDestroyArray(br);
	matClose(pw);
	return hostErrNoError;
}

hostError_t writeFileInt(int *data, int rows, int cols, const char *path, const char *var)
{
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL) return hostErrFileOpenFailed;
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j*rows + i] = (int)data[i*cols + j];
		}
	}
	int status = matPutVariable(pw, var, br);
	if (status != 0)
	{
		//printf("%s", "save result failed");
		return hostErrError;
	}
	mxDestroyArray(br);
	matClose(pw);
	return hostErrNoError;
}

void convert2Matrix(mxArray *ar, float *data)
{
	int rows = mxGetM(ar);
	int cols = mxGetN(ar);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{

			data[i*cols + j] = mxGetPr(ar)[rows*j + i];
		}
	}
}

void matrix_trans(float *data, float *data_trans, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			data_trans[j * m + i] = data[i * n + j];
		}
	}
}

void matrix_multi(float *pxp, float *p1, float *p2, int m, int n, int l)
{
	float alpha = 1.0;
	float beta = 0.0;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, l, n, alpha, p1, n, p2, l, beta, pxp, l);
}

void mean_col(float *p, float *p1, int m, int n)
{
	for (int i = 0; i < n; i++)
	{
		float sum = 0.0;
		for (int j = 0; j < m; j++)
		{
			sum += p[j * n + i];
		}
		p1[i] = sum / m;
	}
}

void matrix_sum_sub(float *leftMatrix, float *rightMatrix, int m, int n, const char opera)
{
	if (opera == '+')
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				leftMatrix[i * n + j] += rightMatrix[i * n + j];
			}
		}
	}
	else if (opera == '-')
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				leftMatrix[i * n + j] = leftMatrix[i * n + j] - rightMatrix[i * n + j];
			}
		}
	}
}

void repmat(float *ori, float *res, int ori_row, int ori_col, int m, int n)
{
	for (int i = 0; i < ori_row; i++)
	{
		for (int j = 0; j < ori_col; j++)
		{
			for (int l = 0; l < m; l++)
			{
				for (int k = 0; k < n; k++)
				{
					res[i * (ori_col * n) + ori_row * l + k] = ori[i * ori_col + j];
				}
			}
		}
	}
}

void dot_div(float *p, float *p1, int m, int n, int k)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			p[i * n + j] = p1[i * n + j] / k;
		}
	}
}

void pca_(float *X, int m, int n)
{
	float *XT = (float*)malloc(sizeof(float) * m * n);
	float *X_mean = (float*)malloc(sizeof(float) * m);
	float *X_mean_trans = (float*)malloc(sizeof(float) * m);
	float *X_trans = (float*)malloc(sizeof(float) * m * n);
	float *X_rep = (float*)malloc(sizeof(float) * m * n);
	//float *X_sub = (float*)malloc(sizeof(float) * m * n);
	float *sigma1 = (float*)malloc(sizeof(float) * m* m);
	float *sigma = (float*)malloc(sizeof(float) * m* m);
	float *final = (float*)malloc(sizeof(float) * m * n);
	matrix_trans(X, XT, m, n);
	mean_col(XT, X_mean, n, m);
	//writeFile(X_mean, m, 1, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\mean.mat", "mean");
	matrix_trans(X_mean, X_mean_trans, 1, m);
	repmat(X_mean_trans, X_rep, m, 1, 1, n);
	matrix_sum_sub(X, X_rep, m, n, '-');
	//writeFile(X, m, n, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\XT.mat", "X_t");
	matrix_trans(X, X_trans, m, n);
	matrix_multi(sigma1, X, X_trans, m, n, m);
	dot_div(sigma, sigma1, m, m, n);
	//writeFile(sigma, m, m, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\sigma.mat", "sig");
	int *k = (int*)malloc(sizeof(int));
	int *l = (int*)malloc(sizeof(int));
	float *alpha = (float*)malloc(sizeof(float)*m);
	float *beta = (float*)malloc(sizeof(float)*m);
	float *u = (float*)malloc(sizeof(float) * m * m);
	float *v = (float*)malloc(sizeof(float) * m * m);
	float *q = (float*)malloc(sizeof(float) * m * m);
	int *iwork = (int*)malloc(sizeof(int) * m);
	lapack_int Info;
	Info = LAPACKE_sggsvd3(LAPACK_ROW_MAJOR, 'U', 'N', 'N', m, m, m,
		k, l, sigma, m, sigma, m, alpha, beta, u, m, v, m, q, m, iwork);
	printf("The returned value is %d\n", Info);
	printf("k = %d\nl = %d\n", k[0], l[0]);
	matrix_multi(final, X_trans, u, n, m, m);
	float *res = (float*)malloc(sizeof(float) * 3 * n);
	int dim = 3;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < dim; j++) {
			res[i * dim + j] = final[i * m + j];
		}
	}
	writeFile(res, n, dim, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\res.mat", "res");
	//writeFile(final, n, m, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\final.mat", "f_score");
	//writeFile(u, m, m, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\u.mat", "u");
	//writeFile(alpha, m, 1, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\v.mat", "alpha");
	//writeFileInt(iwork, m, 1, "D:\\Nothing\\lab_517\\dataTest\\pca_test\\iwork.mat", "iwork");
	
}

int main()
{
	const char *file = NULL;
	//file = "D:\\Nothing\\RX_C\\RX_Matlab\\hydice.mat";
	file = "D:\\Nothing\\RX_C\\RX_Matlab\\hydice\\bowl1.mat";
	matrix *CDataS = (matrix*)malloc(sizeof(matrix));
	readFile(file, "X", CDataS);
	float *CData = CDataS->data;
	int m = CDataS->rows;
	int n = CDataS->cols;
	pca_(CData, m, n);
	getchar();
	return 0;
}