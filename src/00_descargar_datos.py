"""Etapa 0 · Descarga del dataset publicado por Liu et al. (2025) en figshare (CC BY 4.0).

DOI del dataset: 10.6084/m9.figshare.26098048.v1  (archivo data_xlsx.xlsx, ~74 MB).
Si la descarga automática falla, bájalo a mano desde https://doi.org/10.6084/m9.figshare.26098048.v1
y guárdalo como data/data_xlsx.xlsx.
"""
import urllib.request

import pandas as pd

from config import DATA_CSV, DATA_XLSX

URL = "https://ndownloader.figshare.com/files/47382490"


def main():
    DATA_XLSX.parent.mkdir(exist_ok=True)
    if not DATA_XLSX.exists():
        print("Descargando dataset de figshare...")
        urllib.request.urlretrieve(URL, DATA_XLSX)
    print("Convirtiendo a CSV (tarda 1-2 minutos)...")
    df = pd.read_excel(DATA_XLSX, engine="openpyxl")
    df.to_csv(DATA_CSV, index=False)
    print(f"Listo: {df.shape[0]} filas, {df.shape[1]} columnas -> {DATA_CSV}")


if __name__ == "__main__":
    main()
