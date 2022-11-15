#include <iostream>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "auxiliares.cpp"
#include <Eigen/SparseLU>

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
using Eigen::DecompositionOptions;
using Eigen::SimplicialLLT;
using Eigen::UpLoType;
using Eigen::VectorXd;

pair<VectorXd, VectorXd> gauss_seidel(SpMat, VectorXd, int, VectorXd, VectorXd);
pair<VectorXd, VectorXd> jacobi(SpMat, VectorXd, int, VectorXd, VectorXd);
void normalizar(VectorXd, int);

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

    cout << "esta es la D!!!!!!" << endl;
    cout << D << endl;

    cout << "esta es la L!!!!!!" << endl;
    cout << L << endl;

    DL = D - L;

    cout << "esta es la DL!!!!!!" << endl;
    cout << DL << endl;

    // CALCULO DE LA INVERSA DE D - L.-
    SparseLU<SpMat> solver;
    // DL.makeCompressed();
    solver.compute(DL);
    SpMat I(A.cols(), A.cols());
    I.setIdentity(); // ok
    DL_inv = solver.solve(I);

    if (solver.info() != Success)
    {
        cout << "Oh: Very bad" << endl;
    }

    cout << DL_inv << endl;
    cout << U << endl;

    cout << "" << endl;
    // Calculo de la matriz de iteración T y de c:
    T = DL_inv * U;
    cout << T << endl;
    cout << "" << endl;

    c = DL_inv * b;

    xi = x_ini;
    VectorXd error(reps);
    for (int i = 0; i < reps; i++)
    {
        xi = T * xi + c;
        cout << xi << endl;
        cout << "" << endl;

        error[i] = (xi - x_direct).norm();
    }
    normalizar(xi, A.cols());

    return make_pair(xi, error);
}

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
    cout << T << endl;
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
