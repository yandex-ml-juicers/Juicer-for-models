# HISTORY

Журнал экспериментов. Одна строка — один значимый запуск (не обязательно только смерженные в `dev`). 

Колонка "Конфиг" — путь к пресету в `configs/experiment/`, если запуск через него делался; иначе — перечисление ключевых configs.

| Дата | Ветка / PR | Teacher | Student | Метод дистилляции | Конфиг | Метрика | Заметки |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | `exp/resnet50-to-resnet18-baseline` | ResNet-50 | ResNet-18 | vanilla KD (T=4, alpha=0.7) | `configs/experiment/example_resnet50_to_resnet18.yaml` | top-1 acc 68.2% (student) vs 76.1% (teacher) | Baseline для сравнения с feature matching |
| 2026-07-11 | `exp/resnet56-to-resnet20-baseline` | ResNet-56 | ResNet-20 | vanilla KD (T=4, alpha=0.9) | | Student accuracy: 78.57% → Student with KD: 80.60% (+2.03%) | 20 epochs |
| 2026-07-12 | `exp/resnet50-to-resnet18-baseline` | ResNet-50 | ResNet-18 | feature-based Knowledge Distillation(T = 4 alpha1 = 1.0, alpha2 = 1.0, alpha3 = 1.0) |  | Student accuracy: 85.34% -> Student with KD: 87.91%(+2.57%) | 10 epochs

<!--
Шаблон новой строки:
| YYYY-MM-DD | `branch-name` или `#PR` | Teacher | Student | Метод | конфиг/override | метрика | заметки |
-->
