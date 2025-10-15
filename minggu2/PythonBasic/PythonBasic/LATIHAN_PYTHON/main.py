# cara mengakses package yang dibuat, copy dan paste code dalam file main.py

import os

os.chdir(
    r"D:/UrusanKuliah/Perkuliahan/Semester_7/STKI/minggu2/PythonBasic/PythonBasic/LATIHAN_PYTHON"
)

import latihan_package.alpha as a, latihan_package.beta as b

a.alphaSatu()
b.betaSatu()
