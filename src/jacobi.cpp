#include <iostream>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
using Eigen::DecompositionOptions;
using Eigen::SimplicialLLT;
using Eigen::UpLoType;
using Eigen::VectorXd;

pair<VectorXd, VectorXd> jacobi(SpMat, VectorXd, int, VectorXd, VectorXd);
void normalizar(VectorXd, int);

pair<VectorXd, VectorXd> jacobi(SpMat A, VectorXd b, int reps, VectorXd x_ini, VectorXd x_direct)
{
    SpMat D, L, U, TS, TI, T, D_inv;
    VectorXd c, xi;

    // Triangular superior e inferior de A.-
    TS = A.triangularView<UpLoType::Upper>();
    TI = A.triangularView<UpLoType::Lower>();

    // Cálculo de las matrices D, L, U.-
    D = TS.triangularView<UpLoType::Lower>();
    L = -TI + D;
    U = -TS + D;

    // CALCULO DE LA INVERSA DE D.-
    SimplicialLLT<SpMat> solver;
    D.makeCompressed();
    solver.compute(D);
    SpMat I(A.cols(), A.cols());
    I.setIdentity();
    D_inv = solver.solve(I);

    // Calculo de la matriz de iteración T y de c:
    T = D_inv * (L + U);
    c = D_inv * b;

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

void normalizar(VectorXd x, int n)
{
    double sum = 0.00;
    for (int i = 0; i < n; i++)
    {
        sum += x[i];
    }
    for (int j = 0; j < n; j++)
    {
        x[j] /= sum;
    }
}