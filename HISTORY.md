# HISTORY

Журнал экспериментов. Одна строка — один значимый запуск (не обязательно только смерженные в `dev`). 

Колонка "Конфиг" — путь к пресету в `configs/experiment/`, если запуск через него делался; иначе — перечисление ключевых configs.

| Дата | Ветка / PR | Teacher | Student | Метод дистилляции | Конфиг | Метрика | Заметки |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | `exp/resnet50-to-resnet18-baseline` | ResNet-50 | ResNet-18 | vanilla KD (T=4, alpha=0.7) | `configs/experiment/example_resnet50_to_resnet18.yaml` | top-1 acc 68.2% (student) vs 76.1% (teacher) | Baseline для сравнения с feature matching |

<!--
Шаблон новой строки:
| YYYY-MM-DD | `branch-name` или `#PR` | Teacher | Student | Метод | конфиг/override | метрика | заметки |
-->
