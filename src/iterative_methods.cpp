#include <iostream>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "auxiliares.cpp"
#include <Eigen/SparseLU>
#include <cmath>

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
using Eigen::DecompositionOptions;
using Eigen::SimplicialLLT;
using Eigen::UpLoType;
using Eigen::VectorXd;
typedef Eigen::Triplet<double> T; // TINI

double summation(SpMat, VectorXd, int, int, int);
pair<VectorXd, VectorXd> gauss_seidel(SpMat, VectorXd, int, VectorXd, VectorXd, double);
pair<VectorXd, VectorXd> jacobi(SpMat, VectorXd, int, VectorXd, VectorXd, double);
VectorXd normalizar(VectorXd, int);
bool sonIguales(double a, double b);
VectorXd eg(SpMat A, VectorXd b);

pair<VectorXd, VectorXd> gauss_seidel(SpMat A, VectorXd b, int reps, VectorXd x_ini, VectorXd x_direct, double eps)
{
    // Declaración de variables:
    SpMat TS, TI, D, L, U, DL, T, DL_inv;
    VectorXd c, xi;

    int n = A.cols();
    xi = x_ini;
    VectorXd error(reps);
    double err = 1;

    for (int k = 0; k < reps; k++)
    {
        for (int i = 0; i < n; i++)
        {
            xi[i] = (1 / A.coeff(i, i)) * (b[i] - summation(A, xi, i, 0, i - 1) - summation(A, xi, i, i + 1, n - 1));
        }
        error[k] = (normalizar(xi, A.cols()) - x_direct).norm();
        err = error[k];
        if (err <= eps){
            k = reps;
        }
    }

    xi = normalizar(xi, A.cols());

    return make_pair(xi, error);
}

pair<VectorXd, VectorXd> jacobi(SpMat A, VectorXd b, int reps, VectorXd x_ini, VectorXd x_direct, double eps)
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
    double err = 1;

    for (int i = 0; i < reps; i++)
    {
        xi = T * xi + c;
        error[i] = (normalizar(xi, A.cols()) - x_direct).norm();
        err = error[i];
        if (err <= eps){
            i = reps;
        }
    }
    xi = normalizar(xi, A.cols());

    return make_pair(xi, error);
}

// ELIMINACION GAUSSIANA

bool sonIguales(double a, double b)
{
    return abs(a - b) < 1.0e-6;
}

VectorXd eg(SpMat A, VectorXd b)
{
    int n = A.cols();

    // for (int pivot = 0; pivot < n - 1; pivot++){
    //     cout << "A.outerSize() " << A.outerSize() << endl;

    //     double coeff_pivot = A.coeff(pivot, pivot);
    //     for (int k = pivot; k < A.outerSize(); ++k){
    //         for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
    //         {
    //             if(it.col() == pivot){

    //             }

    //             if (it.row() > pivot){
    //                 cout << "valor: " << it.value() << endl;
    //                 cout << "row: " << it.row() << endl;
    //                 cout << "col: " << it.col() << endl;
    //                 cout << "  " << endl;
    //                 it.valueRef(it.value() - ...);
    //             }
                
    //         }
    //     }
    // }

    for (int pivot = 0; pivot < n - 1; pivot++)
    {
        //cout << "PIVOT: " << pivot << endl;
        double coeff_pivot = A.coeff(pivot, pivot);
        
        for (int fila = pivot + 1; fila < n; fila++)
        {
            // valor del pivot: (valor a "anular") / a_(pivot)
            double valor_pivot = A.coeff(fila, pivot) / coeff_pivot;

            b[fila] = b[fila] - (b[pivot] * valor_pivot);

            // Se actualizan todos los valores de la fila con el valor_pivot
            for (int columna = pivot; columna < n; columna++) // Este ciclo tarda mucho
            {
                double val_2 = A.coeff(pivot, columna) * valor_pivot;
                if (!sonIguales(val_2, 0.00))
                {
                    A.coeffRef(fila, columna) -= val_2;
                }
            }
        }
    }

    // Resolución del sistema triangular Ar = b
    VectorXd result(n);

    for (int i = n - 1; i >= 0; i--)
    {
        double acum_suma = 0.00;
        for (int j = n - 1; j > i; j--)
        {
            acum_suma += A.coeff(i, j) * result[j];
        }
        result[i] = (b[i] - acum_suma) / (double)A.coeff(i, i);
    }

    result = normalizar(result, n);

    return result;
}

/*
Funcion de Gauss Seidel que también funciona (implementación vieja).-

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
    xi = normalizar(xi, A.cols());

    return make_pair(xi, error);
}
*/
