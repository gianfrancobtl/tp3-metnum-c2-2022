#include <iostream>
#include <random>
#include <Eigen/Sparse>

using namespace std;

typedef Eigen::SparseMatrix<float> SpMat; // declares a column-major sparse matrix type of double
using Eigen::DecompositionOptions;
using Eigen::SimplicialLLT;
using Eigen::UpLoType;
using Eigen::VectorXd;

// pair<VectorXd, VectorXd> jacobi(SpMat A, VectorXd b, int &reps, VectorXd x_ini, VectorXd x_direct);

pair<VectorXd, VectorXd> jacobi(SpMat A, VectorXd b, int &reps, VectorXd x_ini, VectorXd x_direct)
{
    SpMat D, L, U, TS, TI, T;

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
    SpMat D_inv = solver.solve(I);

    // Calculo de la matriz de iteración T y de c:
    T = D_inv * (L + U);
    VectorXd c = D_inv * b;

    // std::cout << D_inv << std::endl;
    // std::cout << L << std::endl;
    // std::cout << U << std::endl;
    // std::cout << T << std::endl;

    VectorXd xi = x_ini;
    VectorXd error(reps);
    for (int i = 0; i < reps; i++)
    {
        xi = T * xi + c;
        error[i] = (xi - x_direct).norm();
    }

    return make_pair(xi, error);
}