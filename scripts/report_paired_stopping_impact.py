"""Render a reproducible human-readable comparison of three pinned diagnostic runs."""
from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
from typing import Any

import compare_srim_source_overlap as io
from run_paired_stopping_impact import EXPECTED, FLAGS, SHIELDS, SOURCE_FILES


def index_report(report: dict[str, Any], source_grid: int, upper_grid: int) -> dict[tuple[Any, ...], Any]:
    if (report.get("schema_version") != "radar.paired_protected_external_spectra.v1"
            or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in FLAGS.items())
            or report.get("source_reference_sha256") != SOURCE_FILES[source_grid][1]
            or report.get("source_refinement") != source_grid or report.get("upper_grid_points_per_decade_setting") != upper_grid
            or report.get("case_count") != 84 or report.get("source_count") != 14):
        raise ValueError("Wrong experiment status, refinement or source identity.")
    cases = report["cases"]
    indexed = {(r["source"], r["mode"], r["z"], r["thickness_g_cm2"]): r for r in cases}
    if len(cases) != 84 or set(indexed) != {(*key, d) for key in EXPECTED for d in SHIELDS}:
        raise ValueError("Incomplete or duplicate pilot cases.")
    return indexed


def number(x: float) -> str:
    if not isfinite(x):
        raise ValueError("Nonfinite report value.")
    return f"{x:.6g}"


def render(base: dict[str, Any], dense: dict[str, Any], coarse: dict[str, Any], pins: tuple[str, ...]) -> str:
    a, b, c = index_report(base, 1, 80), index_report(dense, 2, 80), index_report(coarse, 1, 40)
    for field in ("controls_sha256", "implementation_sha256", "production_and_source_sha256", "energy_thresholds",
                "let_thresholds", "energy_edges_mev_per_source_A", "let_edges_mev_cm2_mg"):
        io.same(base[field], dense[field], "base/dense "+field)
        io.same(base[field], coarse[field], "base/coarse "+field)
    lines = ["# Влияние выбора тормозных способностей на защищённые спектры",
        "", "Диагностика `INCOMPLETE`: внешние спектры, не орбитальный расчёт и не разрешение на замену таблиц. "
        "2026 год, САС 5 лет, средняя активность, вероятность превышения СКЛ 0,1. "
        "14 входных спектров × 6 защит; U в таблице СКЛ отсутствует.", "",
        "Буквы обозначают модель в Al, затем в Si: S — SRIM, C — CATIMA. "
        "SS, CS, SC, CC рассчитаны на одном интервале **входных** энергий. "
        "В графе «выше SRIM» приведён отдельный расчёт только CATIMA для ионов, **вошедших** с энергией выше последнего узла SRIM; "
        "их конечная энергия может оказаться ниже этого узла. Экстраполяции SRIM нет.", "",
        "Интегралы при PEAK в см⁻²·с⁻¹, при CUMULATIVE в см⁻² за САС. Порог энергии в МэВ/нуклон, "
        "порог ЛПЭ в МэВ·см²/мг. Защита в г/см². `—` означает неопределённый процент при SS=0, не ноль.", "",
        "ΔAl = CS−SS, ΔSi = SC−SS, взаимодействие = CC−CS−SC+SS. "
        "*п.п. входа* означает 100·Δ / интеграл всего входного спектра данного иона, источника и режима. "
        "Удвоение сетки источника и сгущение сетки CATIMA выше SRIM проверены отдельно: "
        "это численные проверки, не границы физической погрешности.", "",
        "SHA256 отчётов (базовый / сгущённый источник / более редкая верхняя сетка CATIMA):", "", *[f"- `{p}`" for p in pins]]
    lines += ["", "## Пример: пиковые ГКЛ за 10 г/см² алюминия", "",
              "Все значения здесь — проценты от **полного входного потока соответствующего иона**. "
              "Для каждого иона выбран свой порог; эти строки нельзя складывать между собой.", "",
              "| Ион | ЛПЭ ≥ | SS | CS | SC | CC | выше SRIM, только C | ниже нижнего узла после Al: S / C |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for z, symbol, threshold in ((2, "He", .01), (10, "Ne", .2), (26, "Fe", 10.), (92, "U", 20.)):
        case = a["gcr", "peak", z, 10.]
        scale = 100 / case["source_integral"]
        row = [symbol, str(threshold), *[number(case["variants"][v]["let_ge"][str(threshold)]*scale)
               for v in ("SS", "CS", "SC", "CC")], number(case["upper_catima_only"]["let_ge"][str(threshold)]*scale),
               " / ".join(number(case["variants"][v]["unresolved_low_residual_integral"]*scale) for v in ("SS", "CC"))]
        lines.append("| "+" | ".join(row)+" |")
    lines += ["", "Последняя колонка относится к общей входной области. Это не установленная доля физически остановившихся частиц.",
              "", "## Все пороги и защиты", ""]
    for key in sorted(a):
        x, y, w = a[key], b[key], c[key]
        io.same(x["variants"], w["variants"], "common variants cannot depend on the upper grid")
        total = x["source_integral"]
        lines.extend(["", f"### {key[0].upper()} · {key[1]} · Z={key[2]} · {key[3]:g} г/см²", "",
            f"Интеграл входа: {number(total)}. Разбиение по входной энергии: " + ", ".join(
               f"{k}={number(v)}" for k, v in x["input_partition"].items()) + ".", "",
            "| Порог | SS | CS | SC | CC | выше SRIM, только C | ΔAl, п.п. входа | ΔSi, п.п. входа | взаимодействие, п.п. входа | ΔCC, % SS | уточнение источника¹ | уточнение верхней сетки² |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for axis, label in (("energy_ge", "E ≥"), ("let_ge", "ЛПЭ ≥")):
            for t in sorted(x["factorial"][axis], key=float):
                d = x["factorial"][axis][t]
                row = [label+" "+t, *[number(x["variants"][v][axis][t]) for v in ("SS", "CS", "SC", "CC")],
                    number(x["upper_catima_only"][axis][t]),
                    *[number(d["percent_of_full_source"][v]) for v in ("Al", "Si", "interaction")],
                    "—" if d["percent_of_SS"]["combined"] is None else number(d["percent_of_SS"]["combined"]),
                    number(100*(y["variants"]["CC"][axis][t]-x["variants"]["CC"][axis][t])/total),
                    number(100*(x["upper_catima_only"][axis][t]-w["upper_catima_only"][axis][t])/total)]
                lines.append("| "+" | ".join(row)+" |")
    lines.extend(["", "¹ Изменение интеграла CC общей области при удвоении плотности входной сетки, п.п. базового входа. "
        "² Изменение вклада выше границы SRIM при переходе с 40 на 80 узлов CATIMA на декаду, п.п. базового входа.", "",
        "Полные спектры по выходным бинам и остаточные интегралы доступны в машинных отчётах. "
        "Неподдержанная остаточная энергия не считается физической остановкой; частицы вне конечной сетки источника не оцениваются."])
    return "\n".join(lines)+"\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "dense", "coarse"):
        p.add_argument("--"+key, type=Path, required=True)
        p.add_argument("--"+key+"-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pins = (args.base_sha256, args.dense_sha256, args.coarse_sha256)
    report = render(io.read_pinned(args.base, pins[0]), io.read_pinned(args.dense, pins[1]),
                    io.read_pinned(args.coarse, pins[2]), pins)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
