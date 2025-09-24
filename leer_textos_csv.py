"""Herramientas para leer columnas de texto desde archivos CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List


def leer_textos_csv(ruta: str | Path, columna: str, *, encoding: str = "utf-8") -> List[str]:
    """Lee los valores de una columna de texto en un archivo CSV.

    Args:
        ruta: Ruta al archivo CSV.
        columna: Nombre de la columna que contiene el texto.
        encoding: Codificación del archivo, por defecto ``"utf-8"``.

    Returns:
        Una lista con el contenido de la columna indicada, sin espacios adicionales.

    Raises:
        FileNotFoundError: Si el archivo CSV no existe.
        ValueError: Si la columna solicitada no se encuentra en el archivo.
    """

    ruta = Path(ruta)
    with ruta.open(newline="", encoding=encoding) as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames is None:
            raise ValueError("El archivo CSV debe contener encabezados.")

        if columna not in lector.fieldnames:
            columnas_disponibles = ", ".join(lector.fieldnames)
            raise ValueError(
                f"La columna '{columna}' no existe en el CSV. "
                f"Columnas disponibles: {columnas_disponibles}"
            )

        textos: List[str] = []
        for fila in lector:
            valor = fila.get(columna, "")
            textos.append(valor.strip())

    return textos


def _crear_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lee una columna de texto desde un archivo CSV y la imprime en pantalla."
    )
    parser.add_argument("ruta", type=Path, help="Ruta al archivo CSV")
    parser.add_argument(
        "columna",
        type=str,
        help="Nombre de la columna que contiene el texto (debe existir en el encabezado).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Codificación del archivo. Por defecto se usa utf-8.",
    )
    return parser


def main(args: Iterable[str] | None = None) -> None:
    parser = _crear_argument_parser()
    parametros = parser.parse_args(args)
    textos = leer_textos_csv(parametros.ruta, parametros.columna, encoding=parametros.encoding)
    for indice, texto in enumerate(textos, start=1):
        print(f"{indice}: {texto}")


if __name__ == "__main__":
    main()
