#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;

// El programa requiere 2 parametros: un archivo txt y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    // Argumentos : 1 - main - 2 archivo entrada - 3 reps
    if (argc != 2)
    {
        cout << "Error: debe ingresar un parámetro" << endl;
        return 1;
    }

    int n = std::stoi(argv[1]);

    ofstream archivoSalida;
    archivoSalida.setf(ios::fixed, ios::floatfield);
    archivoSalida.precision(10);

    // Seteo los nombres de los archivos a guardar.-
    string archivoSalida_c = "../../tests/sumidero_";
    archivoSalida_c += argv[1];
    archivoSalida_c += "_nodos.out";

    archivoSalida.open(archivoSalida_c);
    archivoSalida << n << endl;
    archivoSalida << n - 1 << endl;
    for (int i = 2; i <= n; i++)
    {
        archivoSalida << i << ' ' << '1' << endl;
    }
    archivoSalida.close();
}