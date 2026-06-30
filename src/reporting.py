"""Utilidades de reporte visual para el notebook."""

from __future__ import annotations

import pandas as pd
from IPython.display import HTML, display


def display_scrollable_dataframe(
    df: pd.DataFrame,
    height_px: int = 360,
    title: str | None = None,
) -> None:
    """Muestra un DataFrame en una ventana HTML con scroll.

    Args:
        df: DataFrame a mostrar.
        height_px: Altura máxima de la ventana.
        title: Título opcional de la tabla.
    """
    if df.empty:
        print("El DataFrame está vacío.")
        return

    html_table = df.to_html(
        index=False,
        escape=False,
        classes="scrollable-df-table",
        border=0,
        justify="center",
    )
    title_html = f'<div class="scrollable-df-title">{title}</div>' if title else ""
    html = f"""
    <style>
    .scrollable-df-title {{
        font-family: Arial, sans-serif;
        font-size: 12pt;
        font-weight: bold;
        margin: 10px 0 6px 0;
        color: #222222;
    }}
    .scrollable-df-container {{
        max-height: {height_px}px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #b8b8b8;
        border-radius: 6px;
        background-color: #ffffff;
        padding: 0;
        margin: 8px 0 16px 0;
    }}
    .scrollable-df-table {{
        border-collapse: collapse;
        width: max-content;
        min-width: 100%;
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.35;
        color: #222222;
        background-color: #ffffff;
    }}
    .scrollable-df-table thead th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background-color: #e9eef5;
        color: #111111;
        font-weight: bold;
        border-bottom: 1px solid #aeb7c2;
        padding: 7px 10px;
        white-space: nowrap;
        text-align: center;
    }}
    .scrollable-df-table tbody td {{
        padding: 6px 10px;
        border-bottom: 1px solid #e3e3e3;
        white-space: nowrap;
        text-align: center;
        color: #222222;
    }}
    .scrollable-df-table tbody tr:nth-child(odd) {{
        background-color: #ffffff;
    }}
    .scrollable-df-table tbody tr:nth-child(even) {{
        background-color: #f6f8fb;
    }}
    .scrollable-df-table tbody tr:hover {{
        background-color: #edf4ff;
    }}
    </style>
    {title_html}
    <div class="scrollable-df-container">
        {html_table}
    </div>
    """
    display(HTML(html))


def display_dataframe_blocks(
    df: pd.DataFrame,
    block_size: int = 10,
    height_px: int = 260,
    title_prefix: str = "Bloque",
) -> None:
    """Despliega un dataset en bloques consecutivos con ventanas de 10 registros.

    Args:
        df: DataFrame original.
        block_size: Tamaño de cada bloque.
        height_px: Altura de cada ventana.
        title_prefix: Prefijo textual del título.
    """
    if block_size < 1:
        raise ValueError("block_size debe ser mayor o igual que 1.")

    for start in range(0, len(df), block_size):
        end = min(start + block_size, len(df))
        block = df.iloc[start:end].copy()
        display_scrollable_dataframe(
            block,
            height_px=height_px,
            title=f"{title_prefix}: registros {start + 1} a {end}",
        )
