#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

// Generate random LOWER matrix
void Gen_rand_symm_mat(const int m, const int n, double* A)
{
	srand(20200409);
	// srand(time(NULL));

	// #pragma omp parallel for
	for (int j=0; j<n; j++)
		for (int i=j; i<m; i++)
				A[i+j*m] = A[j+i*m] = 1.0 - 2*(double)rand() / RAND_MAX;
}

void Gen_rand_lower_mat(const int m, const int n, double* A)
{
	srand(20200409);
	// srand(time(NULL));

	// #pragma omp parallel for
	for (int j=0; j<n; j++)
		for (int i=j; i<m; i++)
				A[i+j*m] = 1.0 - 2*(double)rand() / RAND_MAX;
}

void Gen_Hilbert_mat(const int n, double* A)
{
	for (int j=0; j<n; j++)
		for (int i=0; i<n; i++)
			A[i+j*n] = 1.0 / (((double)(i)+1.0)+((double)(j)+1.0) -1.0);
}

void Gen_Hilbert_lower_mat(const int n, double* A)
{
	for (int j=0; j<n; j++)
		for (int i=j; i<n; i++)
			A[i+j*n] = 1.0 / (((double)(i)+1.0)+((double)(j)+1.0) -1.0);
}

void Gen_Yokota_symm_mat(const int n, double* A, const double k)
{
	srand(20200409);
	// srand(time(NULL));

	for (int j=0; j<n; j++)
		for (int i=0; i<n; i++)
			A[i+j*n] = 1.0 / pow(fabs((double)(i) - (double)(j)) + 1.0, k);
}

void Gen_Yokota_lower_mat(const int n, double* A)
{
	srand(20200409);
	// srand(time(NULL));

	for (int j=0; j<n; j++)
		for (int i=j; i<n; i++)
			A[i+j*n] = 1.0 / (fabs((double)(i) - (double)(j)) + 1.0);
}

// Show matrix
void Show_mat(const int m, const int n, double* A)
{
	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
			printf("% 6.4lf, ",A[i + j*m]);
		cout << endl;
	}
	cout << endl;
}
void Show_mat(const int m, const int n, int* A)
{
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
            printf("%d, ",A[i + j*m]);
        cout << endl;
    }
    cout << endl;
}

// Show tile matrix
void Show_tilemat(const int m, const int n, const int mb, const int nb, double* A)
{
	const int p =  (m % mb == 0) ? m/mb : m/mb+1;   // # tile rows
	const int q =  (n % nb == 0) ? n/nb : n/nb+1;   // # tile columns

    for (int i=0; i<m; i++)
    {
        int ii = i/mb;
        int ib = min(m-ii*mb,mb);
        int ti = i - ii*mb;
        for (int j=0; j<n; j++)
        {
            int jj = j/nb;
            int jb = min(n-jj*nb,nb);
            int tj = j - jj*nb;
            
            printf("% 6.4lf, ",A[m*jj*nb + ii*mb*jb + tj*ib + ti]);
        }
        cout << endl;
    }
    cout << endl;
}

void cm2ccrb(const int m, const int n, const int mb, const int nb, const double* A, double* B)
{
	const int p =  (m % mb == 0) ? m/mb : m/mb+1;   // # tile rows
	const int q =  (n % nb == 0) ? n/nb : n/nb+1;   // # tile columns

   for (int j=0; j<q; j++)
    {
        int jb = min(n-j*nb,nb);
        for (int i=j; i<p; i++)
        {
            int ib = min(m-i*mb,mb);
            const double* Aij = A+(j*nb*m + i*mb);
            double* Bij = B+(j*nb*m + i*mb*jb);

            #pragma omp task depend(in: Aij[0:m*jb]) depend(out: Bij[0:ib*jb])
            {
                #ifdef TRACE
                trace_cpu_start();
                trace_label("Yellow", "Conv.");
                #endif

                for (int jj=0; jj<jb; jj++)
                    for (int ii=0; ii<ib; ii++)
                        Bij[ ii + jj*ib ] = Aij[ ii + jj*m ];

                #ifdef TRACE
                trace_cpu_stop("Yellow");
                #endif
            }
        }
    }
}

void ccrb2cm(const int m, const int n, const int mb, const int nb, const double* B, double* A)
{
	const int p =  (m % mb == 0) ? m/mb : m/mb+1;   // # tile rows
	const int q =  (n % nb == 0) ? n/nb : n/nb+1;   // # tile columns

	for (int j=0; j<q; j++)
    {
        int jb = min(m-j*nb,nb);
        for (int i=j; i<p; i++)
        {
            int ib = min(m-i*mb,mb);
            double* Aij = A+(j*nb*m + i*nb);
            const double* Bij = B+(j*nb*m + i*nb*jb);

            #pragma omp task depend(in: Bij[0:ib*jb]) depend(out: Aij[0:m*jb])
            {
                #ifdef TRACE
                trace_cpu_start();
                trace_label("Violet", "Conv.");
                #endif

                for (int jj=0; jj<jb; jj++)
                    for (int ii=0; ii<ib; ii++)
                        Aij[ ii+jj*m ] = Bij[ ii+jj*ib ];

                #ifdef TRACE
                trace_cpu_stop("Violet");
                #endif
            }
        }
    }
}

// Vanish upper element of the matrix
void Vanish_Upper(const int m, const int n, double* A)
{
    for (int j=1; j<n; j++)
		for (int i=0; i<min(m,j); i++)
			A[i+j*m] = 0.0;
}
