//
// Created by Mycloud on 2020/7/27.
//

#include <vector>
#include <mkl.h>
#include <cmath>
#include <cassert>
#include <iostream>

//#define DEBUG
using namespace std;
//static int cou = 0;
//static int pp = 0;
void Show_mat(const int m, const int n, double* A, int lda)
{
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
            printf("% 6.4lf, ",A[i + j*lda]);
        cout << endl;
    }
    cout << endl;
}

void Gen_Upper(const int n, double* A) {
    for (int j=1; j<n; j++)
        for (int i=0; i<j; i++) {
            A[i+j*n] = A[j+i*n];
        }
}

//extract the i th row of A, i starts from 0
double* ex_row(const double *A, const int n, const int i) {
    double* v = new double[n];
    for(int j = 0; j < n; ++j) {
        v[j] = *(A+i+j*n);
    }
    return v;
}

double* ex_col(const double *A, const int n, const int i) {
    double* u = new double[n];
    for(int j = 0; j < n; ++j) {
        u[j] = *(A+i*n+j);
    }
    return u;
}

//extract absolute max value of vector, return the value and index
double ex_pivot(double* a, const int n, vector<int> index) {
    int max = 0;
    double max_val =  (double) INT8_MIN;
    int k = index.size();
    bool flag = false;

    for(int i = 0; i < n; ++i) {
        flag = false;
        for(int j = 0; j < k; ++j) {
            if(i == index[j]) {
                flag = true;
                break;
            }
        }
        if(abs(a[i]) > max_val && !flag) {
            max = i;
            max_val = abs(a[i]);
        }
    }
    return max;
}


void div(double *a, const int n, double pivot) {
    for(int i = 0; i < n; ++i) {
        a[i] /= pivot;
    }
}

//overload constant*array
double* dot_mul(const double *a, const int n, const double k){
    double* res = new double[n];
    for(int i = 0; i < n; ++i) {
        res[i] = k * a[i];
    }
    return res;
}

//overload array substraction
double* dot_sub (const double *a, const double* b, const int n) {
    double* res = new double[n];
    for(int i = 0; i < n; ++i) {
        res[i] = a[i] - b[i];
    }
    return res;
}

//calculate Euclidean norm
double norm_2( const double *a, const int n) {

    double norm = 0;
    for(int i = 0; i < n; ++i) {
        norm += a[i]*a[i];
    }
    return sqrt(norm);
}

//merge the results
void copy(const double *a, const int n, double *X) {
    for(int i = 0; i < n; ++i) {
        *(X+i) = a[i];
    }
}

void aca(const double *A, const int n, double* X, double* Y, double thr, int &rank, int max_rank) {
    //double timer = omp_get_wtime();
    //X*YT is the approximation result of A
    rank = 1;
    int pivot;
    vector<int> col_pivots, row_pivots{0};
    double *u = NULL;
    double *v = NULL;

    v = ex_row(A, n, 0); //first row //step 1
    pivot = ex_pivot(v, n, col_pivots);  //max value of the vector, //step 3
    div(v, n, v[pivot]);    //step 4 this order is wrong
    copy(v, n, Y); // merge to Y
    col_pivots.push_back(pivot);


    u = ex_col(A, n, pivot);
    copy(u, n, X);    //  merge to X
    pivot = ex_pivot(u, n, row_pivots); // pivot must be one
    row_pivots.push_back(pivot);

    double nu = norm_2(u, n) * norm_2(v, n);
    double mu2 = nu * nu;
    double temp = 0;
    while(nu > thr*sqrt(mu2) && rank < max_rank) {

        v = ex_row(A, n, pivot);
        for(int k = 0; k < rank; ++k) {
            v = dot_sub(v, dot_mul(ex_col(Y, n, k), n, ex_col(X,n,k)[pivot]), n);
        }//ite step 1


        pivot = ex_pivot(v, n, col_pivots);  // ite step 2
        col_pivots.push_back(pivot);
        div(v, n, v[pivot]);    //ite step 3

        copy(v, n, Y+rank*n);   //merge to Y

        u = ex_col(A, n, pivot);    //ite step 4
        for(int k = 0; k < rank; ++k) {
            u = dot_sub(u, dot_mul(ex_col(X, n, k), n, ex_col(Y, n, k)[pivot]), n);
        }   //ite step 4

        copy(u, n, X+rank*n);     // merge to X

        temp = 0;

        for(int k = 0; k < rank; ++k) {
            temp += abs(cblas_ddot(n, ex_col(X, n, k), 1, u, 1)) * abs(cblas_ddot(n, ex_col(Y, n, k), 1, v, 1));

        }//ite step 6

        nu = norm_2(u, n) * norm_2(v, n);

        mu2 += 2*temp + nu*nu;

        pivot = ex_pivot(u, n, row_pivots);
        row_pivots.push_back(pivot);
        rank ++;
    }
    //cou ++;
//
//    if(rank > 30) {
//        cout << "block number: " << cou << ", ACA BLOCK TIME: " << omp_get_wtime() - timer << ", rank : " << rank
//             << endl;
//        cout << "count: " << ++pp << endl;
//    }
//debug mode

#ifdef DEBUG
    double* T = new double[n*n];
    cblas_dcopy(n*n, A, 1, T, 1);
if(rank < max_rank) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, rank, -1.0, X, n, Y, n, 1.0, T, n);
    //Show_mat(n, n, T, n);
    cout << "rank: " << rank << ", ACA accuracy: " << cblas_dnrm2(n * n, T, 1) << endl;
}
    delete [] T;
#endif

}

//int main() {
//
//    int n = 20;
//    int rank = 0;
//    double* A = new double [n*n];
//    Gen_Hilbert_mat(n, A);
//    Gen_Upper(n, A);
//    double tau = 1.0e-5;
//    double* X = new double [n*n];
//    double* Y = new double [n*n];
//    aca(A, n, X, Y, tau, rank);
//    Show_mat(n, n, A, n);
//    Show_mat(n, n, X, n);
//    Show_mat(n, n, Y, n);
//    cout << "rank: " << rank << endl;
//}

