
import os

# Get the list of all files and directories
path = input("Caminho da pasta com os arquivos: ")
files = os.listdir(path)


for file in files:
    file_extencion = file.split(".")[-1]
    # if (file_extencion.lower() == "jpg"):
    if (file_extencion.lower() == "webp"):
        os.remove(f"{path}/{file}")
