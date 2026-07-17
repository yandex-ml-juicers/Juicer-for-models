# Juicer-for-models


## Установка

```bash
pip install -r requirements.txt
pip install -e . --no-deps        # пакет src/ становится импортируемым
# для тестов: pip install -e ".[dev]" --no-build-isolation
```

## Быстрый старт

```bash
# Бейзлайн 1: ванильная дистилляция Хинтона, ResNet-56 -> ResNet-20
python scripts/train.py experiment=b1_vanilla_kd

# Контроль к нему: тот же ученик с нуля, без учителя
python scripts/train.py experiment=b1_scratch

# Бейзлайн 2: feature-based KD, ResNet-50 -> ResNet-18 (логиты + карты признаков)
python scripts/train.py experiment=b2_feature_kd
python scripts/train.py experiment=b2_scratch

# Любой гиперпараметр меняется из CLI, без правки кода:
python scripts/train.py experiment=b1_vanilla_kd loss.temperature=8 trainer.epochs=30 seed=1

# Смоук-тест пайплайна: синтетические данные, CPU, без сети, ~10 секунд
python scripts/train.py data=fake '~model/teacher' loss=ce \
    trainer.epochs=1 trainer.limit_train_batches=3 trainer.limit_eval_batches=2

# Оценка сохранённого чекпоинта
python scripts/eval.py experiment=b2_feature_kd ckpt_path=outputs/<name>/<run>/best.pt
```

Артефакты каждого запуска — в `outputs/<name>/<дата_время>/`:
`.hydra/config.yaml` (полный снапшот конфига), `train.log`, `history.csv`
(метрики и все компоненты лосса по эпохам), `best.pt` / `last.pt`.
