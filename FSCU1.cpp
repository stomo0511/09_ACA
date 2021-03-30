//
// Created by Mycloud on 2020/8/15.
//

#include <iostream>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <omp.h>
#include <mkl.h>
#include <fstream>

#include "utils.hpp"
#include "ACA.hpp"

using namespace std;

//#define OUTPUT
// Serial LDLT factorization
//#define CFACT
//#define TRACE
#define BUT_CON

#ifdef TRACE
extern void trace_cpu_start();
extern void trace_cpu_stop(const char *color);
extern void trace_label(const char *color, const char *label);
#endif

void dsytrf(const int m, const int lda, double* A)
{
    double* v = new double [m];
    for (int k=0; k<m; k++)
    {
        for (int i=0; i<k; i++)
            v[i] = A[k+i*lda]*A[i+i*lda];

        v[k] = A[k+k*lda] - cblas_ddot(k,A+k,lda,v,1);
        A[k+k*lda] = v[k];

        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    m-k-1, k, -1.0, A+(k+1), lda, v, 1, 1.0, A+(k+1)+k*lda,1);
        cblas_dscal(m-k-1, 1.0/v[k], A+(k+1)+k*lda, 1);
    }
    delete [] v;
}

// Generate random vector for Butterfly transformation
void Gen_RBT_vec(const int d, const int n, double* w)
{
    // #pragma omp parallel for
    for (int i=0; i<d*n; i++)
    {
        w[i] = exp( (0.5 - (double)rand() / RAND_MAX) / 10.0);
    }
}

// Randomize kernel C = U^T * C * U
void Rand_twoside(const int n, const double* w, const double* v, double* C, const int lda)
{
    // Make matrix D
#pragma omp for
    for (int j=0; j<n/2; j++)
    {
        for (int i=0; i<n/2; i++)
        {
            double c11 = C[i+j*lda];
            double c12 = C[i+(j+n/2)*lda];
            double c21 = C[(i+n/2)+j*lda];
            double c22 = C[(i+n/2)+(j+n/2)*lda];

            // D_11 = C_11 + C_22 + C_21 + C_12
            C[ i     + j*lda     ] = (w[i]/2.0)*(c11 + c22 + c21 + c12)*v[j];
            // D_12 = C_11 - C_22 + C_21 - C_12
            C[ i     +(j+n/2)*lda] = (w[i]/2.0)*(c11 - c22 + c21 - c12)*v[j+n/2];
            // D_21 = C_11 - C_22 - C_21 + C_12
            C[(i+n/2)+ j*lda     ] = (w[i+n/2]/2.0)*(c11 - c22 - c21 + c12)*v[j];
            // D_22 = C_11 + C_22 - C_21 - C_12
            C[(i+n/2)+(j+n/2)*lda] = (w[i+n/2]/2.0)*(c11 + c22 - c21 - c12)*v[j+n/2];
        }
    }
}

// Apply randomization to the matrix
void Apply_Rand2Mat(const int d, const int n, const double* r, double* C, const int lda)
{
    //#pragma omp parallel
    {
        // d=2
        Rand_twoside(n/2, r+n, r+n,         C,                 lda );   // D_11
        Rand_twoside(n/2, r+n, r+n+n/2,     C+(n/2)*lda,       lda );   // D_12
        Rand_twoside(n/2, r+n+n/2, r+n,     C+(n/2),           lda );   // D_21
        Rand_twoside(n/2, r+n+n/2, r+n+n/2, C+(n/2)+(n/2)*lda, lda );   // D_22

        // d=1
        Rand_twoside(n, r, r, C, lda);
    }
}

// Randomize kernel x = U^T * x
void Rand_LeftTrans(const int n, const double* w, double* x)
{
#pragma omp for
    for (int i=0; i<n/2; i++)
    {
        double tmp;
        tmp      = w[i]*    (x[i] + x[i+n/2]) / sqrt(2.0);
        x[i+n/2] = w[i+n/2]*(x[i] - x[i+n/2]) / sqrt(2.0);
        x[i] = tmp;
    }
}

// Apply randomization to the vector
void Apply_TransRand2Vec(const int d, const int n, const double* r, double* x)
{
    //#pragma omp parallel
    {
        // x = U^T_2 * x
        Rand_LeftTrans(n/2, r+n,     x);
        Rand_LeftTrans(n/2, r+n+n/2, x+n/2);

        // x = U^T_1 * x
        Rand_LeftTrans(n, r, x);
    }
}

// Randomize kernel x = U * x
void Rand_LeftNotrans(const int n, const double* w, double* x)
{
#pragma omp for
    for (int i=0; i<n/2; i++)
    {
        double tmp;
        tmp      = (w[i]*x[i] + w[i+n/2]*x[i+n/2]) / sqrt(2.0);
        x[i+n/2] = (w[i]*x[i] - w[i+n/2]*x[i+n/2]) / sqrt(2.0);
        x[i] = tmp;
    }
}

// Apply randomization to the vector
void Apply_NotransRand2Vec(const int d, const int n, const double* r, double* x)
{
#pragma omp parallel
    {
        // x = U_1 * x
        Rand_LeftNotrans(n, r, x);

        // x = U_2 * x
        Rand_LeftNotrans(n/2, r+n,     x);
        Rand_LeftNotrans(n/2, r+n+n/2, x+n/2);
    }
}



int main(const int argc, const char **argv)
{
    /////////////////////////////////////////////////////////
    // Usage "a.out [size of matrix: m ] [tile size: b]"
    if (argc < 3)
    {
        cerr << "usage: a.out[size of matrix: n ] [tile size: nb]\n";
        return EXIT_FAILURE;
    }

    const int m = atoi(argv[1]);
    const int n = (int)pow(2.0,m);       // size of matrix n = 2^m
    const int nb = atoi(argv[2]);        // tile size
    const int p =  (n % nb == 0) ? n/nb : n/nb+1;   // # tiles
    //const string BT = argv[3];      //0: block version, 1: tile version, others error
    cout << "Version: " << argv[3] << endl;
    const int thr = atoi(argv[4]);
    const int rank_thr = atoi(argv[5]);
    const double val_k = atof(argv[6]);
    const int max_rank = nb/rank_thr;        // determine if the block has low-rank property
    const double tau = pow(10.0, -1.0*thr);                // Singular value threshold
    cout << "k = " << val_k << endl;
#ifdef OUTPUT
    fstream fileStream;
    const char * filename = argv[7];
    fileStream.open(filename, ios::app);
    fileStream << "---------------program start------------------" << endl;
    fileStream << "-------Version:  " << argv[3] << " -----------" << endl;
    fileStream << "size: 2^" << m << ", block size: "<< nb << ", threshold: " << tau << ", rank threshold: " << rank_thr << ", k: " << val_k <<endl;
    fileStream << endl;
#endif

    const int d = 2;                     // depth of random butterfly matrix
    double* Rd = new double [n*d];       // random number vectors

    double* A = new double [n*n];        // Original matrix
    double* O = new double [n*n];        // copy of oririnal matrix
    double* G = new double [n*n];        // copy of oririnal matrix for Lapack routine
    double* F = new double [n*n];        // Tiled matrix
    double* OF = new double [n*n];       // Copy of original tile matrix
    double* D = new double [n];          // D_k = diagonal elements of L_{kk}
    const int lda = n;                   // Leading dimension of A

    double* LD = new double [nb*n];      // LD_k = L_{ik}*D_{kk}
    const int ldd = nb;                  // Leading dimension of LD

    double* X = new double [p*nb*p*nb];  // Rectangular matrix
    double* Y = new double [p*nb*p*nb];  // Rectangular matrix//
    int* R = new int [p*p];              // R_{ij} = Rank of F_{ij}


    double* b = new double [n];          // RHS vector
    double* x = new double [n];          // Solution vector
    double* r = new double [n];          // Residure vector

    double t = omp_get_wtime();

    //////////////////////////////////////////////////////////////////////////////////
    t = omp_get_wtime();
    Gen_RBT_vec(d,n,Rd);                 // Generate random number vector

    Gen_Yokota_symm_mat(n,A,val_k);            // Generate Yokota matrix
    cblas_dcopy(n*n,A,1,G,1);
    cblas_dcopy(n*n,A,1,O,1);            // Copy A to O
#ifdef BUT_CON
    Apply_Rand2Mat(d, n, Rd, O, lda);    // Apply RBT to A: O = U^T A U
    Vanish_Upper(n,n,O);
#endif
    cm2ccrb(n,n,nb,nb,O,F);              // Convert CM to CCRB
    cblas_dcopy(n*n, F, 1, OF, 1);       // Copy tile mat F to OF

    cblas_dcopy(n*n,A,1,O,1);            // Copy A to O
    //////////////////////////////////////////////////////////////////////////////////
    cout << "init time: " << omp_get_wtime() - t << endl;
    //////LAPACKE ROUTINE///////////////////
    for (int i=0; i<n; i++)
        b[i] = x[i] = 1.0;

    int* piv = new int [n];

    double timer = omp_get_wtime();          // Timer start
//    double error = 0;

    assert(0 == LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L', n, G, lda, piv));
    assert(0 == LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', n, 1, G, lda, piv, x, lda));

    timer = omp_get_wtime() - timer;  // Timer stop
    cout << n << ", " << "LAPACKE ROUTINE: " << timer << endl;

    // b := A*x - b
    cblas_dsymv(CblasColMajor, CblasLower, n, -1.0, A, lda, x, 1, 1.0, b, 1);
    double error = cblas_dnrm2 (n, b, 1);
    cout << "||b - A*x||_2 = " << error << " (by LAPACK dsytrf)\n\n";
    
#ifdef OUTPUT
    fileStream << "LAPACK DSYTRF: " << timer << "s" << endl;
    fileStream << "||b - A*x||_2 = " << error << " (by LAPACK dsytrf)\n\n" << endl;
    fileStream << endl;
#endif

    ////////////////////////


    timer = omp_get_wtime();      // Timer start

#pragma omp parallel
    {
        double ittimer = omp_get_wtime();
#ifdef BUT_CON
        Apply_Rand2Mat(d, n, Rd, O, lda);        // Apply RBT to A: O = U^T A U
        Vanish_Upper(n,n,O);
#endif
#pragma omp single
        {
            cm2ccrb(n,n,nb,nb,O,F);              // Convert CM to CCRB
            cout << "prepare time: " << omp_get_wtime() - ittimer << endl;
            /////////////////////////////////////////////////////
            // Main loop
            for (int k=0; k<p; k++)
            {
                const int kb = min(n-k*nb,nb);
                double* Fkk = F+(k*nb*kb + k*nb*lda);
                double* Dk = D+k*nb;

                /////////////////////////////////////////////////////
                // Factor: Fkk
#pragma omp task \
					depend(inout: Fkk[0:kb*kb]) \
					depend(out: Dk[0:kb])
                {
#ifdef TRACE
                    trace_cpu_start();
					trace_label("Red", "Factor");
#endif
                    dsytrf(kb,kb,Fkk);               // DSYTRF
                    for (int l=0; l<kb; l++)         // Dk = diag(Fkk)
                        Dk[l] = Fkk[l+l*kb];
                    //cout << "dsytrf" << endl;
#ifdef TRACE
                    trace_cpu_stop("Red");
#endif
                }

                for (int i=k+1; i<p; i++)
                {
                    const int ib = min(n-i*nb,nb);
                    double* Fik = F+(i*nb*kb + k*nb*lda);

                    /////////////////////////////////////////////////
                    // Solve: Fik
#pragma omp task \
						depend(in: Fkk[0:kb*kb], Dk[0:kb]) \
						depend(inout: Fik[0:ib*kb])
                    {
#ifdef TRACE
                        trace_cpu_start();
						trace_label("Cyan", "Solve");
#endif
                        // TRSM: F_{ik} <- F_{ik} L_{kk}^T
                        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                                    ib, kb, 1.0, Fkk, kb, Fik, ib);
                        //cout << "dtrsm" << endl;
                        // F_{ik} <- F_{ik} D_{k}^{-1}
                        for (int l=0; l<kb; l++)
                            cblas_dscal(ib, 1.0/Dk[l], Fik+l*ib, 1);
#ifdef TRACE
                        trace_cpu_stop("Cyan");
#endif
                    }
                }

                for (int i=k+1; i<p; i++)
                {
                    const int ib = min(n-i*nb,nb);

                    double* Fik = F+(i*nb*kb + k*nb*lda);
                    double* Xik = X+(i*nb*nb + k*nb*p*nb);   // Left singular vectors
                    double* Yik = Y+(i*nb*nb + k*nb*p*nb);   // Right singular vectors

                    /////////////////////////////////////////////////
                    // Compress: Fik
#pragma omp task \
						depend(in: Fik[0:ib*kb]) \
						depend(out: Xik[0:nb*nb], Yik[0:nb*nb], R[i+k*p])
                    {
#ifdef TRACE
                        trace_cpu_start();
					    trace_label("Green", "Compress");
#endif

                        int rank = 0;
                        aca(Fik, ib, Xik, Yik, tau, rank, max_rank);
                        R[i + k * p] = rank;

#ifdef TRACE
                        trace_cpu_stop("Green");
#endif
                    }
                }

                for (int i=k+1; i<p; i++)
                {
                    const int ib = min(n-i*nb,nb);
                    double* Fik = F+(i*nb*kb + k*nb*lda);
                    double* Xik = X+(i*nb*nb + k*nb*p*nb);   // Left singular vectors
                    double* Yik = Y+(i*nb*nb + k*nb*p*nb);   // Right singular vectors
                    double* LDk = LD+(k*ldd*ldd);            // LD_k = Xik * Yik^T * Dk

                    /////////////////////////////////////////////////
                    // Update: Fij <- Xik * (Yik^T * Dk * Yjk) * X_jk^T
#pragma omp task \
						depend(in: Dk[0:nb], Xik[0:nb*nb], Yik[0:nb*nb], R[i+k*p], Fik[0:nb*nb]) \
						depend(out: LDk[0:nb*nb])
                    {
                        const int rik = R[i+k*p];
#ifdef TRACE
                        trace_cpu_start();
			            trace_label("Blue", "Update 1");
#endif

                        if(rik < max_rank)
                        {
                            //LDK^T = Dkk*Yik(nb*rik)
                            cblas_dcopy(nb * rik, Yik, 1, LDk, 1);
                            for (int l = 0; l < nb; l++)
                                cblas_dscal(rik, Dk[l], LDk + l, nb);
                        }
//                        else{
//                            cout << "i, k = " << i << "," << k << endl;
//                            cblas_dcopy(ib*kb, Fik, 1, LDk, 1);
//                            for (int l=0; l<kb; l++)
//                                cblas_dscal(rik, Dk[l], LDk+l*nb, 1);
//                        }


#ifdef TRACE
                        trace_cpu_stop("Blue");
#endif
                    }

                    for (int j=k+1; j<=i; j++)
                    {
                        const int jb = min(n-j*nb,nb);
                        double* Fij = F+(i*nb*jb + j*nb*lda);
                        double* Fjk = F+(j*nb*jb + k*nb*lda);
                        double* Xjk = X+(j*nb*nb + k*nb*p*nb);
                        double* Yjk = Y+(j*nb*nb + k*nb*p*nb);

#pragma omp task \
							depend(in: LDk[0:nb*nb], Xjk[0:nb*nb], Yjk[0:nb*nb], R[j+k*p],R[i+k*p], Fjk[0:nb*nb]) \
							depend(inout: Fij[0:ib*jb])
                        {
#ifdef TRACE
                            trace_cpu_start();
							trace_label("Blue", "Update");
#endif
                            const int rjk = R[j+k*p];
                            const int rik = R[i+k*p];
                            double* W1 = new double [nb*nb];
                            double* W2 = new double [nb*nb];


                            if(rik < max_rank && rjk < max_rank) {
//                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
//                                            jb, kb, rjk, 1.0, Xjk, nb, Yjk, nb, 0.0, W, nb);
                                //result W1: rik*rjk = LDk^T * Yjk
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                            rik, rjk, nb, 1.0, LDk, nb, Yjk, nb, 0.0, W1, nb);
                                //result W2:nb*rjk = Xik*W1
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            nb, rjk, rik, 1.0, Xik, nb, W1, nb, 0.0, W2, nb);
                                // nb*nb Fij -= W2*Xjk^T
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            ib, jb, rjk, -1.0, W2, nb, Xjk, nb, 1.0, Fij, ib);
                            }
                            else {
                                cblas_dcopy(ib*kb, Fik, 1, W1, 1);
                                for (int l=0; l<kb; l++)
                                    cblas_dscal(ib, Dk[l], W1+l*nb, 1);
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            ib, jb, nb, -1.0, W1, nb, Fjk, nb, 1.0, Fij, ib);
                            }
//                            else{
//                                cblas_dcopy(jb*kb, Fjk, 1, W2, 1);
//                                // Fij <- Fij - LD * W^T
//                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
//                                            ib, jb, kb, -1.0, LDk, nb, W2, nb, 1.0, Fij, ib);
//                            }
                            // cout << "update 2" << endl;



                            // Vanish upper part of Fii
                            if (i==j)
                                Vanish_Upper(ib, jb, Fij);

#ifdef TRACE
                            trace_cpu_stop("Blue");
#endif

                            delete [] W1;
                            delete [] W2;
                        }
                    } // End of j-loop
                } // End of i-loop
            } // End of k-loop
        } // End of single region
    } // End of parallel region

    timer = omp_get_wtime() - timer; // Timer stop
    // cout << n << ", " << timer << endl;
    cout << "size, program time: " <<  n << ", " << timer << ", " << endl;
#ifdef OUTPUT
    fileStream << "program time: " << timer << "s" << endl;
    fileStream << endl;
#endif
    //Show_mat(p, p, R);
    timer = omp_get_wtime();         // Timer start
    //////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////
    // Set the diagonal elements of F to 1
    for (int k=0; k<p; k++)
    {
        const int kb = min(n-k*nb,nb);
        double* Fkk = F+(k*nb*kb + k*nb*lda);
        for (int l=0; l<kb; l++)
            Fkk[l+l*kb] = 1.0;
    }

    //////////////////////////////////////////////////////////////////
    // Check || b - (LDLt)*x ||
    for (int i=0; i<n; i++)
        b[i] = x[i] = 1.0;
#ifdef BUT_CON
    Apply_TransRand2Vec(d, n, Rd, x);  // x := U^T b
#endif
    // Solve L*x = b for x
    for (int k=0; k<p; k++)
    {
        const int kb = min(n-k*nb,nb);
        double* Fkk = F+(k*nb*kb + k*nb*lda);
        double* xk = x + k*nb;

        // Solve F_{kk} x_k = b_k for x_k
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    kb, 1, 1.0, Fkk, kb, xk, kb);

        // Update x_i
        for (int i=k+1; i<p; i++)
        {
            const int ib = min(n-i*nb,nb);
            double* Fik = F+(i*nb*nb + k*nb*lda);
            double* Xik = X+(i*nb*nb + k*nb*p*nb);
            double* Yik = Y+(i*nb*nb + k*nb*p*nb);
            const int rik = R[i+k*p];
            double* xi = x + i*nb;

            double* W = new double [nb*nb];

            // W = Xik * Yik^T
            if(rik < max_rank)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            ib, kb, rik, 1.0, Xik, nb, Yik, nb, 0.0, W, nb);
            else{
                cblas_dcopy(ib*kb, Fik, 1, W, 1);
            }
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        ib, kb, -1.0, W, nb, xk, 1, 1.0, xi, 1);

            delete [] W;
        }
    }

    // x := D^{-1} x
    for (int i=0; i<n; i++)
        x[i] /= D[i];

    // Solve L^{T}*y = x for y(x)
    for (int k=p-1; k>=0; k--)
    {
        const int kb = min(n-k*nb,nb);
        double* Fkk = F+(k*nb*kb + k*nb*lda);
        double* xk = x + k*nb;

        // Solve Ft_{kk} y_k = x_k for y_k
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                    kb, 1, 1.0, Fkk, kb, xk, kb);

        // Update x_i
#pragma omp parallel for
        for (int i=k-1; i>=0; i--)
        {
            const int ib = min(n-i*nb,nb);
            double* Fki = F+(k*nb*kb + i*nb*lda);
            double* Xki = X+(k*nb*nb + i*nb*p*nb);
            double* Yki = Y+(k*nb*nb + i*nb*p*nb);
            const int rki = R[k+i*p];
            double* xi = x + i*nb;

            double* W = new double [nb*nb];

            if(rki < max_rank)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            kb, ib, rki, 1.0, Xki, nb, Yki, nb, 0.0, W, nb);
            else{
                cblas_dcopy(ib*kb, Fki, 1, W, 1);
            }

            cblas_dgemv(CblasColMajor, CblasTrans,
                        kb, ib, -1.0, W, nb, xk, 1, 1.0, xi, 1);

            delete [] W;
        }
    }
#ifdef BUT_CON
    Apply_NotransRand2Vec(d, n, Rd, x);
#endif
    timer = omp_get_wtime() - timer; // Timer stop
    cout << "||b - LDLT||_2 time: " << timer << ", " << endl;

    // b := b - A*x
    cblas_dsymv(CblasColMajor, CblasLower, n, -1.0, A, lda, x, 1, 1.0, b, 1);
    error = cblas_dnrm2(n, b, 1);
    cout << "No piv LDLT:   || b - (LDLt)*x ||_2 = " << error << endl;
    //cout << cblas_dnrm2(n, b, 1) << ", ";
#ifdef OUTPUT
    fileStream << "No piv LDLT (solved): " << timer << "s" << endl;
    fileStream << "||b - A*x||_2 = " << error << " (No piv LDLT (solved))\n\n" << endl;
    fileStream << endl;
#endif

    timer = omp_get_wtime();         // Timer start

    //////////////////////////////////////////////////////////////////
    // Iterative refinement
    cblas_dcopy(n,b,1,r,1);
    for (int i=0; i<n; i++)
        b[i] = 1.0;

    // Solve L*y = r for y(r)
    for (int k=0; k<p; k++)
    {
        const int kb = min(n-k*nb,nb);
        double* Fkk = F+(k*nb*kb + k*nb*lda);
        double* rk = r + k*nb;

        // Solve F_{kk} x_k = r_k for x_k
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    kb, 1, 1.0, Fkk, kb, rk, kb);

        // Update r_i
#pragma omp parallel for
        for (int i=k+1; i<p; i++)
        {
            const int ib = min(n-i*nb,nb);
            double* Fik = F+(i*nb*kb + k*nb*lda);
            double* Xik = X+(i*nb*nb + k*nb*p*nb);
            double* Yik = Y+(i*nb*nb + k*nb*p*nb);
            const int rik = R[i+k*p];
            double* ri = r + i*nb;

            double* W = new double [nb*nb];

            if(rik < max_rank)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            ib, kb, rik, 1.0, Xik, nb, Yik, nb, 0.0, W, nb);
            else{
                cblas_dcopy(ib*kb, Fik, 1, W, 1);
            }

            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        ib, kb, -1.0, W, nb, rk, 1, 1.0, ri, 1);

            delete [] W;
        }
    }

    // r := D^{-1} r
    for (int i=0; i<n; i++)
        r[i] /= D[i];

    // Solbe L^{T}*y = r for y(r)
    for (int k=p-1; k>=0; k--)
    {
        const int kb = min(n-k*nb,nb);
        double* Fkk = F+(k*nb*kb + k*nb*lda);
        double* rk = r + k*nb;

        // Solve Ft_{kk} y_k = r_k for y_k
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                    kb, 1, 1.0, Fkk, kb, rk, kb);

        // Update r_i
#pragma omp parallel for
        for (int i=k-1; i>=0; i--)
        {
            const int ib = min(n-i*nb,nb);
            double* Fki = F+(k*nb*kb + i*nb*lda);
            double* Xki = X+(k*nb*nb + i*nb*p*nb);
            double* Yki = Y+(k*nb*nb + i*nb*p*nb);
            const int rki = R[k+i*p];
            double* ri = r + i*nb;

            double* W = new double [nb*nb];

            // W = Xki * Yki^T

            if(rki < max_rank)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            kb, ib, rki, 1.0, Xki, nb, Yki, nb, 0.0, W, nb);
            else{
                cblas_dcopy(kb*ib, Fki, 1, W, 1);
            }


            cblas_dgemv(CblasColMajor, CblasTrans,
                        kb, ib, -1.0, W, nb, rk, 1, 1.0, ri, 1);

            delete [] W;
        }
    }

    // x := x+r
    cblas_daxpy(n,1.0,r,1,x,1);

    timer = omp_get_wtime() - timer; // Timer stop
    cout << "Iterative time: " << timer << ", " << endl;

    // b := b - A*x
    cblas_dsymv(CblasColMajor, CblasLower, n, -1.0, A, lda, x, 1, 1.0, b, 1);
    error = cblas_dnrm2(n, b, 1);
    cout << "Apply 1 ItRef: || b - (LDLt)*x ||_2 = " << error << endl;
    //cout << cblas_dnrm2(n, b, 1) << endl;
#ifdef OUTPUT
    fileStream << "Iterative time = " << timer << " (by one iterative refinement)\n\n" << endl;
    fileStream << endl;
    fileStream << "||b - (LDLt)*x||_2 = " << error << " (by one iterative refinement)\n\n" << endl;
    fileStream << endl;
#endif

#ifdef CFACT
    ////////////////////////////////////////////////////////////////
    // Check || O - L*D*Lt ||
#pragma omp parallel for
    for (int j=0; j<p; j++)
    {
        const int jb = min(n-j*nb,nb);

        for (int i=j; i<p; i++)
        {
            const int ib = min(n-i*nb,nb);
            double* Aij = OF+(i*nb*jb + j*nb*lda);

            for (int k=0; k<=j; k++)
            {
                const int kb = min(n-k*nb,nb);
                double* Dk = D+k*nb;
                double* Wt = new double [nb*nb];

                // Wt <- Ljk
                if (k==j) // Ljk is dense
                {
                    double* Fjk = F+(j*nb*kb + k*nb*lda);

                    // Wt <- Ljk
                    for (int l=0; l<kb; l++)
                        cblas_dcopy(jb, Fjk+l*jb, 1, Wt+l*nb, 1);
                }
                else      // Ljk is low-rank
                {

                    double* Xjk = X+(j*nb*nb + k*nb*p*nb);
                    double* Yjk = Y+(j*nb*nb + k*nb*p*nb);
                    const int rjk = R[j+k*p];

                    // Wt <- Xjk * Yjk^T

                    if(rjk < max_rank)
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    jb, kb, rjk, 1.0, Xjk, nb, Yjk, nb, 0.0, Wt, nb);
                    else{
                        double* Fjk = F+(j*nb*kb + k*nb*lda);
                        cblas_dcopy(jb*kb, Fjk, 1, Wt, 1);
                    }

                }

                // Wt <- Wt * Dk
                for (int l=0; l<kb; l++)
                    cblas_dscal(jb, Dk[l], Wt+l*nb, 1);

                if (k==i) // Lik is dense
                {
                    double* Fik = F+(i*nb*kb + k*nb*lda);

                    // Wt <- Wt * Ltik
                    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                                jb, kb, 1.0, Fik, ib, Wt, nb);
                    // OFij -= Wt
                    for (int jj=0; jj<jb; jj++)
                        for (int ii=jj; ii<ib; ii++)
                            Aij[ii+jj*ib] -= Wt[ii+jj*nb];
                }
                else      // Lik is low-rank
                {
                    double* Fik = F+(i*nb*nb, k*nb*lda);
                    double* Xik = X+(i*nb*nb + k*nb*p*nb);
                    double* Yik = Y+(i*nb*nb + k*nb*p*nb);
                    const int rik = R[i+k*p];

                    double* W = new double [nb*nb];

                    // W <- Xik * Yik^T
                    if(rik < max_rank)
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    ib, kb, rik, 1.0, Xik, nb, Yik, nb, 0.0, W, nb);
                    else{
                        cblas_dcopy(ib*kb, Fik, 1, W, 1);

                    }

                    // OFij -= W * Wt
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                ib, jb, kb, -1.0, W, nb, Wt, nb, 1.0, Aij, ib);

                    if (i==j)    // vanish upper part of Aij
                    {
                        Vanish_Upper(ib, jb, Aij);
                    }
                    delete [] W;
                }
                delete [] Wt;
            } // End of k-loop
        } // End of i-loop
    } // End of j-loop

    error = cblas_dnrm2(n*n, OF, 1);
    cout << "|| A - LDLt ||_2 = " << error << endl;
#endif
#ifdef OUTPUT
    fileStream << "|| A - LDLt ||_2 = " << error << " (No piv LDLT)\n\n" << endl;
    fileStream << endl;
#endif


    delete [] Rd;
    delete [] A;
    delete [] O;
    delete [] F;
    delete [] OF;
    delete [] D;
    delete [] X;
    delete [] Y;
    delete [] R;
    delete [] b;
    delete [] x;
    delete [] r;

    return EXIT_SUCCESS;
}

