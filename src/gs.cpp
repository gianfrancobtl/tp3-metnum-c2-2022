#include <iostream>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;

typedef Eigen::SparseMatrix<double> SpMat;
using Eigen::DecompositionOptions;
using Eigen::SimplicialLLT;
using Eigen::UpLoType;
using Eigen::VectorXd;

// pair<VectorXd, VectorXd> gauss_seidel(SpMat A, VectorXd b, int &reps, VectorXd x_ini, VectorXd x_direct);
// void normalizar(VectorXd, int);

pair<VectorXd, VectorXd> gauss_seidel(SpMat A, VectorXd b, int reps, VectorXd x_ini, VectorXd x_direct)
{
    // Declaración de variables:
    SpMat TS, TI, D, L, U, DL, T, DL_inv;
    VectorXd c, xi;

    // Triangular superior e inferior de A.-
    TS = A.triangularView<UpLoType::Upper>();
    TI = A.triangularView<UpLoType::Lower>();

    // Cálculo de las matrices D, L, U.-
    D = TS.triangularView<UpLoType::Lower>();
    L = -TI + D;
    U = -TS + D;

    DL = D - L;

    // CALCULO DE LA INVERSA DE D - L.-
    SimplicialLLT<SpMat> solver;
    DL.makeCompressed();
    solver.compute(DL);
    SpMat I(A.cols(), A.cols());
    I.setIdentity();
    DL_inv = solver.solve(I);

    // Calculo de la matriz de iteración T y de c:
    T = DL_inv * U;
    c = DL_inv * b;

    xi = x_ini;
    VectorXd error(reps);
    for (int i = 0; i < reps; i++)
    {
        xi = T * xi + c;
        error[i] = (xi - x_direct).norm();
    }
    normalizar(xi, A.cols());

    return make_pair(xi, error);
}

// void normalizar(VectorXd x, int n)
// {
//     double sum = 0.00;
//     for (int i = 0; i < n; i++)
//     {
//         sum += x[i];
//     }
//     for (int j = 0; j < n; j++)
//     {
//         x[j] /= sum;
//     }
// }