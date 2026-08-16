# Скачивание датасета

1. Зарегестрироваться на сайте cityscapes под вузовской почтой

2. Скачать файлы
```
python -m pip install cityscapesscripts
```
```
csDownload -l
csDownload -d data/raw/cityscapes leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip
```

3. Разархивировать их
```
unzip data/raw/cityscapes/leftImg8bit_trainvaltest.zip -d data/raw/cityscapes
unzip data/raw/cityscapes/gtFine_trainvaltest.zip -d data/raw/cityscapes
```
или
```
python3 -m zipfile -e data/raw/cityscapes/leftImg8bit_trainvaltest.zip data/raw/cityscapes
python3 -m zipfile -e data/raw/cityscapes/gtFine_trainvaltest.zip data/raw/cityscapes
```

4. Запустить скрипт создающий разметку для детекции
```
python3 -m scripts.convert_cityscapes_to_coco --root data/raw/cityscapes
```