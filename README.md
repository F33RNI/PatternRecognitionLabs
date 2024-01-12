# Лабораторные работы по "Распознавание образов"

Какие лабовы готовы:
- 1 - Классификация изображения с использование Keras
- 2 - Трансфертное обучение
- 3 - Повышение точности сверточной нейронной сети

----------

## 🕒 Подготовка к запуску лабов

1. Установить Python версии 3.10 или выше _(возможно будет работать и на Python 3.9)_, если не установлено
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

### Работа 1. Классификация изображения с использование Keras

1. Перейдите в директорию LR1 используя команду `cd LR1`, находясь в основной папке PatternRecognitionLabs с активированной виртуальной средой (см. секцию `🕒 Подготовка к запуску лабов`)
2. В файле `main.py` поменяйте `IMAGE_PATH ` на путь к любой картинке которую хотите распознать
3. Запустите скрипт используя команду `python main.py`. Выход должен выглядеть примерно так:
```
1/1 [==============================] - 2s 2s/step
('n03944341', 'pinwheel', 1.0)
('n03476684', 'hair_slide', 3.3065376e-08)
('n03271574', 'electric_fan', 6.2919483e-09)
```
4. Если всё верно, начните запись видео, изменяйте `IMAGE_PATH ` на разные пути к разным картинкам, запускайте скрипт и показывайте результат
5. Готово

### Работа 2. Трансфертное обучение

1. **Если для скачивания этого репозитория вы использовали `git`**, то для того, чтобы подгрузить новые файлы для этой лабораторной выполните команду `git pull`
2. Активируйте виртуальную среду и установите недостающие пакеты `pip install -r requirements.txt`
3. Перейдите в директорию LR2 используя команду `cd LR2`
4. В файле `main.py` поменяйте `IMAGE_PATH ` на путь к фотке кошбки / шобаки которую хочите классифицировать
   1. Рекомендую использовать примеры из датасета `kagglecatsanddogs_5340`. Например: `kagglecatsanddogs_5340/PetImages/Cat/0.jpg`
   2. _(в датасете, с целью экономии места, занимаемого репозиторием, было оставлено только по 20 картинок)_
5. Запустите скрипт используя команду `python main.py`. Выход должен выглядеть примерно так
```
Predicted Label: cat
Probabilities: tensor([[0.9989, 0.0011]], grad_fn=<SoftmaxBackward0>)
```
6. Если всё верно, начните запись видео, изменяйте `IMAGE_PATH ` на разные пути к разным картинкам, запускайте скрипт и показывайте результат
7. Для большего масштаба, советую на видео показать структуру датасета и сказать что взят он был от сюда https://www.microsoft.com/en-us/download/details.aspx?id=54765. Модель же была взята предобученная. А если что, структура типичной модели cats vs dogs выглядеть может так (это тоже можно показать на видео):
```python
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS)))
model.add(BatchNormalization(axis=-1))
model.add(Activation(activation))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation(activation))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation(activation))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation(activation))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256,init='he_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(Activation(activation))
model.add(Dropout(0.2))

model.add(Dense(256,init='he_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(Activation(activation))
model.add(Dropout(0.2))

model.add(Dense(1,init='he_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
return model
```

### Работа 3.  Повышение точности сверточной нейронной сети

1. **Если для скачивания этого репозитория вы использовали `git`**, то для того, чтобы подгрузить новые файлы для этой лабораторной выполните команду `git pull`
2. Активируйте виртуальную среду и установите недостающие пакеты `pip install -r requirements.txt`
3. Запустите сервер TensorBoard
   1. Откройте второй терминал, перейдите в директорию в директорию LR3 используя команду `cd LR3`
   2. Создайте папку `tensorboard` при помощи команды `mkdir tensorboard`
   3. Запустите сервер TensorBoard, используя команду `tensorboard --logdir ./tensorboard --alsologtostderr`
   4. Откройте TensorBoard в браузере перейдя по указанной в терминале ссылке (обычно http://localhost:6006/)
4. В основном терминале перейдите в директорию LR3 используя команду `cd LR3`
5. Скачайте датасет от сюда https://www.microsoft.com/en-us/download/details.aspx?id=54765 и распакуйте архив куда угодно
6. В файле `main.py` в `DATASET_PATH` укажите путь к папке PetImages внутри распакованного датасета
7. Запустите скрипт используя команду `python main.py` и дождитесь начала обучения (дождитесь чтобы прошла 1 эпоха хотябы)
   1. Если у Вас установлена CUDA и вы хотите обучать, используя GPU, но возникла ошибка, попробуйте указать путь к CUDA таким образом:
   ```shell
   ╰─ which nvcc
   /opt/cuda/bin/nvcc

   XLA_FLAGS=--xla_gpu_cuda_data_dir="/opt/cuda/" python main.py
   ```
   2. Либо же, обучать используя только CPU:
   ```shell
   CUDA_VISIBLE_DEVICES="-1" python main.py
   ```
8. Откройте TensorBoard и покажите на видео как происходит процесс обучения (как в терминале так и в TensorBoard)
9.  Прекратите выполнение обучения спустя какое то время, нажав CTRL+C
10. Включите `EARLY_STOPPING_ENABLED` в `main.py` и запустите обучение заново. По идее, оно должно будет остановиться само спустя какое то время
11. Ещё, можно изменять `LEARNING_RATE`. От этого тоже будет меняться скорость обучения. На видео показывайте графики функции потери и точности предсказаний
12. В общем, изменяйте чо нить и показывайте на видео кучу графиков
