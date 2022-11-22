#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "iterative_methods.cpp"
#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sys/time.h>
#include <random>
#include <Eigen/SparseLU>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
SpMat generarMatrizDesdeArchivo(ifstream &, double);
VectorXd fillRandomVector(int);
SpMat generarD(SpMat, int);
VectorXd generarE(int);
double calcularGrado(SpMat, double);

// El programa requiere 2 parametros: un archivo txt y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    // Inicializacion variables para calcular tiempo:
    timeval startGauss, endGauss;
    long elapsed_mtime;    /* elapsed time in milliseconds */
    long elapsed_seconds;  /* diff between seconds counter */
    long elapsed_useconds; /* diff between microseconds counter */
    double timeGauss, totalTiempos;

    // Inicializacion de otras variables:
    int n, reps;
    SpMat A;
    VectorXd b, x_direct, x_eg, tiempos(5);
    pair<VectorXd, VectorXd> x_jacobi, x_gauss_seidel;
    cout << "Corriendo el programa..." << endl;

    // Argumentos : 1 - main - 2 archivo entrada - 3 reps
    if (argc != 3)
    {
        cout << "Error: faltan argumentos." << endl;
        return 1;
    }

    // Lee archivo entrada:
    ifstream matrizDeEntrada(argv[1]);
    reps = std::stoi(argv[2]);

    // Prepara el archivo salida para su escritura.-
    ofstream Gauss;
    ofstream GaussSeidel;
    ofstream Jacobi;
    ofstream GaussSeidel_error;
    ofstream Jacobi_error;

    // Seteo tipo de salida:
    Gauss.setf(ios::fixed, ios::floatfield);
    GaussSeidel.setf(ios::fixed, ios::floatfield);
    Jacobi.setf(ios::fixed, ios::floatfield);
    GaussSeidel_error.setf(ios::fixed, ios::floatfield);
    Jacobi_error.setf(ios::fixed, ios::floatfield);

    // Seteo cantidad de decimales (10).-
    Gauss.precision(10);
    GaussSeidel.precision(10);
    Jacobi.precision(10);
    GaussSeidel_error.precision(10);
    Jacobi_error.precision(10);

    // Seteo los nombres de los archivos a guardar.-
    string Gauss_c = "./";
    Gauss_c += argv[1];
    string GaussSeidel_c = Gauss_c + "_GaussSeidel.out";
    string Jacobi_c = Gauss_c + "_Jacobi.out";
    string GaussSeidel_error_c = Gauss_c + "_GaussSeidel_error.out";
    string Jacobi_error_c = Gauss_c + "_Jacobi_error.out";
    Gauss_c += "_Gauss.out";

    double timeCreation = 0.00;
    gettimeofday(&startGauss, NULL);

    double p = 0.75;
    A = generarMatrizDesdeArchivo(matrizDeEntrada, p);
    b = generarE(A.cols());
    VectorXd x_ini = fillRandomVector(A.cols());

    gettimeofday(&endGauss, NULL);
    elapsed_seconds = endGauss.tv_sec - startGauss.tv_sec;
    elapsed_useconds = endGauss.tv_usec - startGauss.tv_usec;
    timeCreation += ((elapsed_seconds)*1000 + elapsed_useconds / 1000.0);

    // ELIMINACION GAUSSIANA (X_DIRECT):
    totalTiempos = 0.00;
    for (int iteraciones = 0; iteraciones < 5; iteraciones++)
    {
        timeGauss = 0.00;
        gettimeofday(&startGauss, NULL);

        SparseLU<SpMat, COLAMDOrdering<int> > solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        x_direct = solver.solve(b);
        x_direct = normalizar(x_direct, A.cols());

        gettimeofday(&endGauss, NULL);
        elapsed_seconds = endGauss.tv_sec - startGauss.tv_sec;
        elapsed_useconds = endGauss.tv_usec - startGauss.tv_usec;
        timeGauss = timeCreation + ((elapsed_seconds)*1000 + elapsed_useconds / 1000.0);
        tiempos[iteraciones] = timeGauss + timeCreation;
        totalTiempos += timeGauss;
    }

    Gauss.open(Gauss_c);
    Gauss << totalTiempos / 5 + timeCreation << endl;
    for (int i = 0; i < 5; i ++){
        Gauss << tiempos[i] << endl;
    }
    Gauss << x_direct << endl;
    Gauss.close();

    // JACOBI:
    totalTiempos = 0.00;
    for (int iteraciones = 0; iteraciones < 5; iteraciones++)
    {
        timeGauss = 0.00;
        gettimeofday(&startGauss, NULL);
        x_jacobi = jacobi(A, b, reps, x_ini, x_direct);
        gettimeofday(&endGauss, NULL);
        elapsed_seconds = endGauss.tv_sec - startGauss.tv_sec;
        elapsed_useconds = endGauss.tv_usec - startGauss.tv_usec;
        timeGauss = timeCreation + ((elapsed_seconds)*1000 + elapsed_useconds / 1000.0);
        tiempos[iteraciones] = timeGauss + timeCreation;        
        totalTiempos += timeGauss;
    }
    Jacobi.open(Jacobi_c);
    Jacobi << totalTiempos / 5 + timeCreation << endl;
    for (int i = 0; i < 5; i ++){
        Jacobi << tiempos[i] << endl;
    }
    Jacobi << x_jacobi.first << endl;
    Jacobi.close();

    Jacobi_error.open(Jacobi_error_c);
    Jacobi_error << x_jacobi.second << endl;
    Jacobi_error.close();

    // GAUSS SEIDEL:
    totalTiempos = 0.00;
    for (int iteraciones = 0; iteraciones < 5; iteraciones++)
    {
        timeGauss = 0.00;
        gettimeofday(&startGauss, NULL);
        x_gauss_seidel = gauss_seidel(A, b, reps, x_ini, x_direct);
        gettimeofday(&endGauss, NULL);
        elapsed_seconds = endGauss.tv_sec - startGauss.tv_sec;
        elapsed_useconds = endGauss.tv_usec - startGauss.tv_usec;
        timeGauss = timeCreation + ((elapsed_seconds)*1000 + elapsed_useconds / 1000.0);
        tiempos[iteraciones] = timeGauss  + timeCreation;
        totalTiempos += timeGauss;
    }
    GaussSeidel.open(GaussSeidel_c);
    GaussSeidel << totalTiempos / 5 + timeCreation << endl;
    for (int i = 0; i < 5; i ++){
        GaussSeidel << tiempos[i] << endl;
    }
    GaussSeidel << x_gauss_seidel.first << endl;
    GaussSeidel.close();
    
    GaussSeidel_error.open(GaussSeidel_error_c);
    GaussSeidel_error << x_gauss_seidel.second << endl;
    GaussSeidel_error.close();

    // IMPRESION DE RESULTADOS POR CONSOLA:
    // cout << "Resultado directo por eliminación gaussiana:" << endl;
    // cout << x_direct << endl;
    // cout << "============================" << endl;
    // cout << "Resultado de Jacobi:" << endl;
    // cout << x_jacobi.first << endl;
    // cout << "============================" << endl;
    // cout << "Resultado de Gauss Seidel:" << endl;
    // cout << x_gauss_seidel.first << endl;

    //  Fin de la ejecución
    return 0;
}

SpMat generarMatrizDesdeArchivo(ifstream &archivoDeEntrada, double p)
{
    // Variables n (cantidad de páginas) - k (cantidad de links) - p1 (pagina 1) - p2 (pagina 2).-
    int n, k, p1, p2;
    archivoDeEntrada >> n;
    archivoDeEntrada >> k;
    std::vector<T> tl;
    tl.reserve(k);
    if (archivoDeEntrada.is_open())
    {
        for (int i = 0; i < k; i++)
        {
            archivoDeEntrada >> p1;
            archivoDeEntrada >> p2;
            // Seteo del valor 1 a la posición [p2][p1]:
            tl.push_back(T(p2 - 1, p1 - 1, 1));
        }
    }
    // Nueva instancia del resultado;
    // Se setean la matriz W y la cantidad total de links entre las páginas.
    SpMat W(n, n);
    // Generación de los "ingredientes" de A:
    W.setFromTriplets(tl.begin(), tl.end());
    SpMat I(W.cols(), W.cols());
    I.setIdentity();
    SpMat D = generarD(W, n);
    // Generación de A:
    SpMat A = I - (p * W * D);
    return A;
}

VectorXd fillRandomVector(int n)
{
    VectorXd v(n);
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        double b = rand() % 20 + 1;
        v[i] = b;
    }
    return v;
}

SpMat generarD(SpMat W, int n)
{
    SpMat D(n, n);
    std::vector<T> tl;
    tl.reserve(n);
    for (int j = 0; j < n; j++)
    {
        double valor = calcularGrado(W, j);
        if (valor == 0)
        {
            tl.push_back(T(j, j, 0.00));
        }
        else
        {
            tl.push_back(T(j, j, 1.00 / (double)valor));
        }
    }
    D.setFromTriplets(tl.begin(), tl.end());
    return D;
}

double calcularGrado(SpMat W, double j)
{
    double res = 0.00;
    for (int i = 0; i < W.cols(); i++)
    {
        res += W.coeff(i, j);
    }
    return res;
}

VectorXd generarE(int n)
{
    VectorXd e(n);
    for (int i = 0; i < n; i++)
    {
        e[i] = 1.00;
    }
    return e;
}