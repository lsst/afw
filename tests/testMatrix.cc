// -*- lsst-c++ -*-

#include <boost/numeric/ublas/io.hpp>
#include <boost/timer.hpp> 
#include <boost/numeric/ublas/matrix.hpp>

#include <vw/Math/Matrix.h> 
#include <vw/Math/Vector.h> 
#include <vw/Math/LinearAlgebra.h> 

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_machine.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_linalg.h>

using namespace boost::numeric;
using namespace std;

int main(int argc, char** argv)
{
    int N = atoi(argv[1]);
    int Nmax = 100;

    /* test matrices */
    ublas::matrix<double> bm1(N,N), bm2(N,N);
    vw::Matrix<double>    vm1(N,N), vm2(N,N);
    vw::Vector<double>    vv1(N);
    gsl_matrix *gm1 = gsl_matrix_alloc(N, N);
    gsl_matrix *gm2 = gsl_matrix_alloc(N, N);
    gsl_vector *gv1 = gsl_vector_alloc(N);

    boost::timer t;
    double time;

    /* 
     *
     *
     First test, fill and increment elements of a matrix 
     *
     *
     */

    /* Boost timing */
    t.restart();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (N > Nmax) {
                bm1(i, j) = (N * i + j) * rand();
                bm2(i, j) = (N * i + j) * rand();
            }
            else {
                for (int k = 0; k < N; ++k) {
                    for (int l = 0; l < N; ++l) {
                        bm1(i, j) += (N * l + k) * rand();
                        bm2(i, j) += (N * l + k) * rand();
                    }
                }
            }
        }
    }
    time = t.elapsed();
    cout << "Boost matrix fill : " << time << "s" << endl;

    /* VW timing */
    t.restart();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (N > Nmax) {
                vm1(i, j) = (N * i + j) * rand();
                vm2(i, j) = (N * i + j) * rand();
            }
            else {
                for (int k = 0; k < N; ++k) {
                    for (int l = 0; l < N; ++l) {
                        vm1(i, j) += (N * l + k) * rand();
                        vm2(i, j) += (N * l + k) * rand();
                    }
                }
            }
        }
    }
    time = t.elapsed();
    cout << "VW matrix fill : " << time << "s" << endl;

    /* GSL timing */
    t.restart();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (N > Nmax) {
                gsl_matrix_set(gm1, i, j, (N * i + j) * rand());
                gsl_matrix_set(gm2, i, j, (N * i + j) * rand());
            }
            else {
                for (int k = 0; k < N; ++k) {
                    for (int l = 0; l < N; ++l) {
                        gsl_matrix_set(gm1, i, j,
                                       gsl_matrix_get(gm1, i, j) + (N * l + k) * rand());
                        gsl_matrix_set(gm2, i, j,
                                       gsl_matrix_get(gm2, i, j) + (N * l + k) * rand());
                    }
                }
            }
        }
    }
    time = t.elapsed();
    cout << "GSL matrix fill 1 : " << time << "s" << endl;

    gsl_matrix_set_zero(gm1);
    gsl_matrix_set_zero(gm2);
    t.restart();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (N > Nmax) {
                *gsl_matrix_ptr(gm1, i, j) = (N * i + j) * rand();
                *gsl_matrix_ptr(gm2, i, j) = (N * i + j) * rand();
            }
            else {
                for (int k = 0; k < N; ++k) {
                    for (int l = 0; l < N; ++l) {
                        *gsl_matrix_ptr(gm1, i, j) += (N * l + k) * rand();
                        *gsl_matrix_ptr(gm2, i, j) += (N * l + k) * rand();
                    }
                }
            }
        }
    }
    time = t.elapsed();
    cout << "GSL matrix fill 2 : " << time << "s" << endl;


    /* Stride over columns */
    gsl_matrix_set_zero(gm1);
    gsl_matrix_set_zero(gm2);
    t.restart();
    for (int i = 0; i < N; ++i) {
        gsl_vector_view row1 = gsl_matrix_row(gm1, i);
        gsl_vector_view row2 = gsl_matrix_row(gm2, i);
        for (int j = 0; j < N; ++j) {
            if (N > Nmax) {
                *gsl_vector_ptr(&row1.vector, j) = (N * i + j) * rand();
                *gsl_vector_ptr(&row2.vector, j) = (N * i + j) * rand();
            }
            else {
                for (int k = 0; k < N; ++k) {
                    for (int l = 0; l < N; ++l) {
                        *gsl_vector_ptr(&row1.vector, j) += (N * l + k) * rand();
                        *gsl_vector_ptr(&row2.vector, j) += (N * l + k) * rand();
                    }
                }
            }
        }
    }
    time = t.elapsed();
    cout << "GSL matrix fill 3 : " << time << "s" << endl;

    /* Stride over rows */
    gsl_matrix_set_zero(gm1);
    gsl_matrix_set_zero(gm2);
    t.restart();
    for (int i = 0; i < N; ++i) {
        gsl_vector_view col1 = gsl_matrix_column(gm1, i);
        gsl_vector_view col2 = gsl_matrix_column(gm2, i);
        for (int j = 0; j < N; ++j) {
            if (N > Nmax) {
                *gsl_vector_ptr(&col1.vector, j) = (N * i + j) * rand();
                *gsl_vector_ptr(&col2.vector, j) = (N * i + j) * rand();
            }
            else {
                for (int k = 0; k < N; ++k) {
                    for (int l = 0; l < N; ++l) {
                        *gsl_vector_ptr(&col1.vector, j) += (N * l + k) * rand();
                        *gsl_vector_ptr(&col2.vector, j) += (N * l + k) * rand();
                    }
                }
            }
        }
    }
    time = t.elapsed();
    cout << "GSL matrix fill 4 : " << time << "s" << endl;

    /* 
     *
     *
     Second test, find linear algebra solution using PCA
     *
     *
     */
    cout << endl;

    for (int i = 0; i < N; ++i) {
        vv1(i) = i * rand();
        gsl_vector_set(gv1, i, i * rand());
    }

    /* use vw's internal least squares mechanism that uses SVD */
    t.restart();
    vw::math::Vector<double> vs1 = vw::math::least_squares(vm1, vv1);
    time = t.elapsed();
    cout << "VW least squared 1 : " << time << "s" << " (" << vs1[0] << " " << vs1[N-1] << ")" << endl;

    /* explicitly use pseudoinverse, which also uses SVD */
    t.restart();
    vw::math::Matrix<double> vm1t = vw::math::pseudoinverse(vm1);
    vw::math::Vector<double> vs2  = vm1t * vv1;
    time = t.elapsed();
    cout << "VW least squared 2 : " << time << "s" << " (" << vs2[0] << " " << vs2[N-1] << ")" << endl;
    
    /* use full GSL solution, includes chi2 calculation and covariance matrix from the residuals */
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc (N, N);
    gsl_vector *gs1                     = gsl_vector_alloc (N);
    gsl_matrix *gc1                     = gsl_matrix_alloc (N, N);
    double chi2;
    size_t rank;
    t.restart();
    gsl_multifit_linear_svd(gm1, gv1, GSL_DBL_EPSILON, &rank, gs1, gc1, &chi2, work);
    time = t.elapsed();
    cout << "GSL least squared 1 : " << time << "s" << " (" << gsl_vector_get(gs1, 0) << " " << gsl_vector_get(gs1, N-1) << ")" << endl;

    /* only use the GSL parts necessary to get the solution */
    gsl_vector *gs2 = gsl_vector_alloc (N);
    t.restart();
    gsl_matrix *A   = work->A;
    gsl_matrix *Q   = work->Q;
    gsl_matrix *QSI = work->QSI;
    gsl_vector *S   = work->S;
    gsl_vector *xt  = work->xt;
    gsl_vector *D   = work->D;
    gsl_matrix_memcpy (A, gm1);
    gsl_linalg_balance_columns (A, D);
    gsl_linalg_SV_decomp_mod (A, QSI, Q, S, xt);
    gsl_blas_dgemv (CblasTrans, 1.0, A, gv1, 0.0, xt);
    gsl_matrix_memcpy (QSI, Q);
    {
        double alpha0 = gsl_vector_get (S, 0);
        size_t p_eff = 0;

        const size_t p = gm1->size2;

        for (size_t j = 0; j < p; j++)
        {
            gsl_vector_view column = gsl_matrix_column (QSI, j);
            double alpha = gsl_vector_get (S, j);
            
            if (alpha <= GSL_DBL_EPSILON * alpha0) {
                alpha = 0.0;
            } else {
                alpha = 1.0 / alpha;
                p_eff++;
            }
            
            gsl_vector_scale (&column.vector, alpha);
        }
        
        rank = p_eff;
    }
    gsl_vector_set_zero (gs2);
    gsl_blas_dgemv (CblasNoTrans, 1.0, QSI, xt, 0.0, gs2);
    gsl_vector_div (gs2, D);
    time = t.elapsed();
    cout << "GSL least squared 2 : " << time << "s" << " (" << gsl_vector_get(gs2, 0) << " " << gsl_vector_get(gs2, N-1) << ")" << endl;
   
    return 0;
}
