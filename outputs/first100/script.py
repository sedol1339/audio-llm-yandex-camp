#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

INPUT_CSV = "results.csv"          # ваш исходный CSV
OUTPUT_XLSX = "asr_report.xlsx"    # финальный Excel

def main():
    # --- 1. читаем данные ----------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip() for c in df.columns]      # на всякий случай уберём пробелы

    # --- 2. сводная таблица: строки = model, столбцы = dataset --------------
    pivot = df.pivot_table(index="model",
                           columns="dataset",
                           values="wer",
                           aggfunc="mean")

    # --- 3. агрегаты по моделям ---------------------------------------------
    by_model = (
        df.groupby("model")["wer"]
          .agg(["count", "mean", "median", "std", "min", "max"])
          .sort_values("mean")
    )
    by_model["rank_by_mean"] = by_model["mean"].rank(method="min")   # 1 = лучший

    # сопоставим «человеческие» имена моделей, если они есть
    if "model_model_name" in df.columns:
        names = (df.dropna(subset=["model_model_name"])
                     .groupby("model")["model_model_name"]
                     .first())
        by_model = by_model.merge(names, left_index=True,
                                  right_index=True, how="left")

    # --- 4. пишем в Excel с форматированием ----------------------------------
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        # листы
        df.to_excel(writer,               sheet_name="raw_data", index=False)
        pivot.to_excel(writer,            sheet_name="pivot_model_x_dataset")
        by_model.to_excel(writer,         sheet_name="models_summary")

        workbook  = writer.book
        ws_pivot  = writer.sheets["pivot_model_x_dataset"]

        # диапазон данных в сводной таблице (для условного форматирования)
        nrows, ncols = pivot.shape
        first_row, first_col = 2, 2          # Excel-координаты данных (учитываем заголовки pandas)
        last_row  = first_row + nrows - 1
        last_col  = first_col + ncols - 1

        # вспом. функция для диапазона
        def xl_range(r1, c1, r2, c2):
            def col_letter(col):
                s = ''
                while col >= 0:
                    s = chr(col % 26 + ord('A')) + s
                    col = col // 26 - 1
                return s
            return f"{col_letter(c1)}{r1+1}:{col_letter(c2)}{r2+1}"

        # 4-а) градиент (зелёный → жёлтый → красный)
        data_rng = xl_range(first_row-1, first_col-1, last_row-1, last_col-1)
        ws_pivot.conditional_format(data_rng, {
            'type':      '3_color_scale',
            'min_color': "#63BE7B",
            'mid_color': "#FFEB84",
            'max_color': "#F8696B"
        })

        # 4-б) подсветка минимального WER в каждом столбце (лучшая модель)
        for c in range(first_col-1, last_col):
            col_rng = xl_range(first_row-1, c, last_row-1, c)
            ws_pivot.conditional_format(col_rng, {
                'type':     'formula',
                'criteria': f'INDIRECT("R"&MATCH(MIN({col_rng}),{col_rng},0)+{first_row-1}&"C{c+1}",FALSE)=INDIRECT("R"&ROW()&"C{c+1}",FALSE)',
                'format':   workbook.add_format({'bold': True, 'font_color': 'white',
                                                 'bg_color': '#4F81BD'})
            })

        # 4-в) диаграмма «средний WER по моделям» на листе summary
        ws_models = writer.sheets["models_summary"]
        chart = workbook.add_chart({'type': 'column'})

        mean_col_idx = by_model.columns.get_loc("mean")  # 0-based в pandas
        start_row = 1
        end_row   = start_row + len(by_model) - 1

        chart.add_series({
            'name':       'Средний WER',
            'categories': ["models_summary", start_row, 0, end_row, 0],  # индекс (модель)
            'values':     ["models_summary", start_row, mean_col_idx+1,
                           end_row, mean_col_idx+1],
            'data_labels': {'value': True, 'num_format': '0.0000'}
        })

        chart.set_title({'name': 'Средний WER по моделям (меньше = лучше)'})
        chart.set_y_axis({'name': 'WER'})
        chart.set_legend({'position': 'bottom'})
        ws_models.insert_chart('J2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

    print(f"Готово! Создан файл {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()