#include <new>
#include <cstdio>
#include <cstdlib>

#include <stdint.h>
#include <stdbool.h>

/*two 32bit integers should be enough for a quite some time in terms
  total possible memory.*/

#define U32MAX 0xFFFFFFFF

namespace Matrix {

typedef struct {
	double *column;
}MATRIX_Row;

class Matrix {
public:
	uint32_t nrows;
	uint32_t ncolumns;
	MATRIX_Row *row;

	~Matrix();
	Matrix(uint32_t irows, uint32_t icolumns);
};

Matrix::~Matrix()
{
	uint32_t i;

	for (i = 0; i < nrows; i++)
		free(row[i].column);

	free(row);
}

Matrix::Matrix(uint32_t irows, uint32_t icolumns)
{
	uint32_t i;

	if (!irows || !icolumns)
		throw std::exception();

	nrows    = irows;
	ncolumns = icolumns;

	row = (MATRIX_Row*)calloc(sizeof(MATRIX_Row), irows);
	if (!row)
		throw std::bad_alloc();

	for (i = 0; i < irows; i++) {
		row[i].column = (double*)calloc(sizeof(double), icolumns);
		if (!row[i].column) {
			while (i--)
				free(row[i].column);
			free(row);
			throw std::bad_alloc();
		}
	}
}

/*The exponential size increasing matrix multiplication*/
Matrix Kronecker(Matrix& left, Matrix& right)
{
	uint64_t limit_test;
	uint32_t m, n, p, q;
	uint32_t m_step, n_step;
	uint32_t n_end, m_end, p_end, q_end;

	/*The overly large and logical limit*/
	limit_test = ((uint64_t)left.nrows)    * ((uint64_t)right.nrows)    *
		     ((uint64_t)left.ncolumns) * ((uint64_t)right.ncolumns) *
		     ((uint64_t)sizeof(double));

	/*Using the multiplicative property of zero, I can check to see if
	 the number of rows and columns in each Matrix are indead greater than
	 zero.*/
	if (!limit_test)
		throw std::exception();
	/*0x3FFFFFFF00000001 is (2^32-1) * (2^32-1)*/
	else if (limit_test > ((uint64_t)0x3FFFFFFF00000001))
		throw std::exception();
	else if (((uint64_t)left.nrows)   * ((uint64_t)right.nrows) > U32MAX ||
		((uint64_t)left.ncolumns) * ((uint64_t)right.ncolumns) > U32MAX)
		throw std::exception();


	m_end = left.nrows;
	n_end = left.ncolumns;
	p_end = right.nrows;
	q_end = right.ncolumns;

	m_step = left.nrows;
	n_step = left.ncolumns;
	Matrix ret(left.nrows * right.nrows, left.ncolumns * right.ncolumns);

	for (m = 0; m < m_end; m++) {
		for (n = 0; n < n_end; n++) {
			for (p = 0; p < p_end; p++) {
				for (q = 0; q < q_end; q++) {
					ret.row[m*m_step+p].column[n*n_step+q] =\
					left.row[m].column[n]                  *\
					right.row[p].column[q];
				}
			}
		}
	}

	return ret;
}

/*Component wise multiplication*/
Matrix Hadamard(Matrix& l, Matrix& r)
{
	uint32_t i, j;

	if (l.nrows != r.nrows || l.ncolumns != r.ncolumns ||
	    l.nrows <= 0 || l.ncolumns <= 0)
		throw std::exception();

	Matrix ret(l.nrows, l.ncolumns);

	for (i = 0; i < ret.nrows; i++)
		for (j = 0; j < ret.ncolumns; j++) {
			ret.row[i].column[j] = \
			l.row[i].column[j]   * \
			r.row[i].column[j];
		}

	return ret;
}

/*horizontal concatenation of matrices*/
Matrix HoriCat(Matrix& l, Matrix& r)
{
	uint32_t i, run;

	if (l.nrows != r.nrows ||
	    l.nrows <= 0 || l.ncolumns <= 0 ||
	    r.nrows <= 0 || r.ncolumns <= 0)
		throw std::exception();

	Matrix ret(l.nrows, l.ncolumns + r.ncolumns);

	for (i = 0; i < l.nrows; i++) {
		for (run = 0; run < l.ncolumns; run++)
			ret.row[i].column[run] = l.row[i].column[run];

		for (run = l.ncolumns; run < l.ncolumns+r.ncolumns; run++)
			ret.row[i].column[run] = r.row[i].column[run-l.ncolumns];
	}

	return ret;
}

/*Dot product*/
/*Normal matrix multiplcation*/
Matrix Mul(Matrix& l, Matrix& r)
{
	uint32_t i, j, k;
	uint32_t common_length;

	if (l.ncolumns != r.nrows ||
	    l.nrows <= 0 || l.ncolumns <= 0 ||
	    r.nrows <= 0 || r.ncolumns <= 0)
		throw std::exception();

	common_length = l.ncolumns;
	Matrix ret(l.nrows, r.ncolumns);

	for (i = 0; i < ret.nrows; i++)
		for (j = 0; j < ret.ncolumns; j++)
			ret.row[i].column[j] = 0.0;

	for (i = 0; i < common_length; i++)
		for (j = 0; j < l.nrows; j++) {
			for (k = 0; k < r.ncolumns; k++) {
				ret.row[j].column[k] += \
				  l.row[j].column[i] *  \
				  r.row[i].column[k];
			}
		}

	return ret;
}

Matrix Add(Matrix& l, Matrix& r)
{
	uint32_t i, j;

	if (l.nrows    != r.nrows    ||
	    l.ncolumns != r.ncolumns ||
	    l.nrows <= 0    || r.nrows <= 0 ||
	    l.ncolumns <= 0 || r.ncolumns <= 0)
		throw std::exception();

	Matrix ret(l.nrows, l.ncolumns);

	for (i = 0; i < l.nrows; i++)
		for (j = 0; j < l.ncolumns; j++)
			ret.row[i].column[j] = \
			  l.row[i].column[j] + \
			  r.row[i].column[j];

	return ret;
}

Matrix Sub(Matrix& l, Matrix& r)
{
	uint32_t i, j;

	if (l.nrows    != r.nrows    ||
	    l.ncolumns != r.ncolumns ||
	    l.nrows <= 0    || r.nrows <= 0 ||
	    l.ncolumns <= 0 || r.ncolumns <= 0)
		throw std::exception();

	Matrix ret(l.nrows, l.ncolumns);

	for (i = 0; i < l.nrows; i++)
		for (j = 0; j < l.ncolumns; j++)
			ret.row[i].column[j] = \
			  l.row[i].column[j] - \
			  r.row[i].column[j];

	return ret;
}

Matrix Transpose(Matrix& m)
{
	uint32_t i, j;

	if (m.ncolumns <= 0 || m.nrows <= 0)
		throw std::exception();

	Matrix ret(m.ncolumns, m.nrows);

	for (i = 0; i < ret.nrows; i++)
		for (j = 0; j < ret.ncolumns; j++)
			ret.row[i].column[j] = \
			  m.row[j].column[i];

	return ret;
}

void MulByScalar(double scalar, Matrix& m)
{
	uint32_t i, j;

	if (m.ncolumns <= 0 || m.nrows == 0)
		throw std::exception();

	for (i = 0; i < m.nrows; i++)
		for (j = 0; j < m.ncolumns; j++)
			m.row[i].column[j] *= scalar;
}

void Invert(double k, Matrix& m)
{
	uint32_t i, j;

	if (m.nrows <= 0 || m.ncolumns <= 0)
		throw std::exception();

	for (i = 0; i < m.nrows; i++)
		for (j = 0; j < m.ncolumns; j++)
			m.row[i].column[j] = k / m.row[i].column[j];
}



} /*namespace Matrix*/

#ifndef NDEBUG
void PrintMatrix(Matrix::Matrix& m)
{
	uint32_t i, j;

	for (i = 0; i < m.nrows; i++) {
		for (j = 0; j < m.ncolumns; j++) {
			printf("%3.0f", m.row[i].column[j]);
			if (j != m.ncolumns-1)
				putchar(' ');
		}
		putchar('\n');
	}
}

int main(int argc, char *argv[])
{
	uint32_t i, j;
	Matrix::Matrix l(2, 2), r(2, 2);

	for (i = 0; i < l.nrows; i++)
		for (j = 0; j < r.ncolumns; j++) {
			l.row[i].column[j] = i+1;
			r.row[i].column[j] = j+5;
		}

	printf("Matrix l:\n");
	PrintMatrix(l);

	printf("Matrix r:\n");
	PrintMatrix(r);

	Matrix::Matrix k = Matrix::Kronecker(l, r);
	printf("KroneckerProduct:\n");
	PrintMatrix(k);

	Matrix::Matrix h = Matrix::Hadamard(l, r);
	printf("HadamardMultiplication:\n");
	PrintMatrix(h);

	Matrix::Matrix hc = Matrix::HoriCat(l, r);
	printf("HorizontalConcatenation:\n");
	PrintMatrix(hc);

	Matrix::Matrix m  = Matrix::Mul(l, r);
	printf("Basic matrix multiplcation:\n");
	PrintMatrix(m);

	Matrix::Matrix a = Matrix::Add(l, r);
	printf("Basic matrix addition:\n");
	PrintMatrix(a);

	Matrix::Matrix s = Matrix::Sub(l, r);
	printf("Basic matrix subtraction:\n");
	PrintMatrix(s);

	Matrix::Matrix t = Matrix::Transpose(l);
	printf("Transposition:\n");
	PrintMatrix(t);

	return 0;
}
#endif

















