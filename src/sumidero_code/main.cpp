#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;

// El programa requiere 2 parametros: un archivo txt y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    // Argumentos : 1 - main - 2 archivo entrada - 3 reps
    if (argc != 3)
    {
        cout << "Error: debe ingresar un parámetro" << endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    int con = std::stoi(argv[2]);

    ofstream archivoSalida;
    archivoSalida.setf(ios::fixed, ios::floatfield);
    archivoSalida.precision(10);

    // Seteo los nombres de los archivos a guardar.-
    string archivoSalida_c = "../../tests/test_sumidero_";
    archivoSalida_c += argv[1];
    archivoSalida_c += "nodos";
    archivoSalida_c += argv[2];
    archivoSalida_c += "_conex.txt";

    archivoSalida.open(archivoSalida_c);
    archivoSalida << n << endl;
    archivoSalida << con << endl;
    for (int i = 2; i <= con+1; i++)
    {
        archivoSalida << i << ' ' << '1' << endl;
    }
    archivoSalida.close();
}