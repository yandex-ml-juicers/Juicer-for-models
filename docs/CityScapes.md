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

3. Запустить скрипт создающий разметку для детекции
```
pyton3 -m scripts.convert_cityscapes_to_coco --root data/raw/cityscapes
```