## DLib install

```
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install 
```

pip install dlib==19.21.1

[Ref](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
[build a custom face recognition dataset](https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/)

https://github.com/feitgemel/Jetson-Nano-Python/blob/master/Install-MediaPipe/Setup.txt

## Face Recognition Usage

### Build dataset
```
python3 build_face_dataset.py --cascade models/haarcascade_frontalface_default.xml --output dataset/Huy
```

### Encode 

```
python3 encode_faces.py --dataset dataset --encodings encodings.pickle
```

### Face Recognize

```
python3 recognize_faces_image.py --encodings encodings.pickle  --image examples/00001.png
python3 recognize_faces_camera.py --encodings encodings.pickle --display 1
```
