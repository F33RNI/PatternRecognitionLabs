# Лабораторные работы по "Распознавание образов"

Какие лабовы готовы:
- 1 - Классификация изображения с использование Keras

----------

## 🕒 Подготовка к запуску лабов

1. Установить Python версии 3.10 или выше, если не установлено
   1. Проверить установку можно командой в терминале `python --version`
   2. Однако, если вы устанавливали Python без автоматического добавления в системные переменные, вам необходимо найти путь к исполняемому файлу. На Windows это обычно `C:\Users\Ваше имя пользователя\AppData\Local\Programs\Python\PythonВЕРСИЯ\python.exe`. **Если это так, в шаге 4 (ТОЛЬКО В ШАГЕ 4), вместо `python` вам необходимо использовать `"C:\Users\Ваше имя пользователя\AppData\Local\Programs\Python\PythonВЕРСИЯ\python.exe"`** 
      1. А чтобы проверить версию:
      ```shell
      "C:\Users\Ваше имя пользователя\AppData\Local\Programs\Python\PythonВЕРСИЯ\python.exe" --version
      ```
2. Скопируйте репозиторий с лабами вручную или используя команду `git clone https://github.com/F33RNI/PatternRecognitionLabs`
3. Откройте терминал и войдите в эту папку используя команду `cd "путь\к папке PatternRecognitionLabs"`
4. Создайте виртуальную среду используя команду `python -m venv venv`
5. Активируйте виртуальную среду. На **Windows**, для этого пропишите в терминале `venv\Scripts\activate.bat`. На **Linux** `source venv/bin/activate`
   1. Если виртуальная среда создана и запущена верно, в терминале перед текущей рабочей директорией появится `(venv)`
6. Установите все пакеты для лабов, используя команду `pip install -r requirements.txt`
7. Пропишите `pip list`. Вы должны увидеть много установленных пакетов, включая `tensorflow`, `numpy`, `keras`, `Pillow` и другие
8. Готово!

----------

## 🏗️ Запуск лабов и создание отчётов

В качестве отчётов вам необходимо будет записать видео (поэтому в этом репозитории нет примеров отчётов). Ниже представлены инструкции по запуску лабов

1. **Работа 1. Классификация изображения с использование Keras**
   1. Перейдите в директорию LR1 используя команду `cd LR1`, находясь в основной папке PatternRecognitionLabs с активированной виртуальной средой (см. секцию `🕒 Подготовка к запуску лабов`)
   2. В файле `main.py` поменяйте `IMAGE_PATH ` на путь к любой картинке которую хотите распознать
   3. Запустите скрипт используя команду `python main.py`. Выход должен выглядеть примерно так
    ```
    1/1 [==============================] - 2s 2s/step
    ('n03944341', 'pinwheel', 1.0)
    ('n03476684', 'hair_slide', 3.3065376e-08)
    ('n03271574', 'electric_fan', 6.2919483e-09)
    ```
   4. Если всё верно, начните запись видео, изменяйте `IMAGE_PATH ` на разные пути к разным картинкам, запускайте скрипт и показывайте результат
   5. Готово
