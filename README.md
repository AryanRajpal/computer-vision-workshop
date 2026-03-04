# CV Workshop — Image Classifier

**OpenCV + GoogLeNet (ImageNet)  ·  Beginner Computer Vision Workshop**

---

## Setup

```bash
pip install opencv-python numpy
```

Then download the model files (one-time, ~50 MB):

```bash
python download_model.py
```

Sample images are already included in `images/`.

---

## Repo Structure

```
cv-workshop/
├── utils.py                    ← Part 1: implement this first
├── model.py                    ← Part 2: implement this second
├── main.py                     ← Part 3: wire everything together
├── download_model.py           ← run once to fetch model files
├── synset_words.txt            ← 1000 ImageNet labels
├── deploy.prototxt             ← model architecture  (after download)
├── bvlc_googlenet.caffemodel   ← model weights       (after download)
├── images/
│   ├── dog.jpg
│   ├── cat.jpg
│   ├── car.jpg
│   └── bird.jpg
```

---

## How to Work Through This

**Work in order: `utils.py` → `model.py` → `main.py`**

Each file has a self-test. Run it after you finish that file:

```bash
python utils.py    # must show all ✓ before moving to model.py
python model.py    # must show all ✓ before moving to main.py
python main.py     # runs the full pipeline
```

---

## Running the Classifier

```bash
# Default image
python main.py

# Specify image and expected label
python main.py --image images/cat.jpg --label cat
python main.py --image images/car.jpg --label "sports car"

# Classify every image in a folder
python main.py --batch images/
```

---

## The Pipeline

```
your_image.jpg
    ↓  load_image()
(H × W × 3)  BGR array
    ↓  preprocess()         grayscale → blur → Canny
(H × W)  binary edge map
    ↓  find_subject_contour()
largest qualifying contour
    ↓  crop_roi()
(h × w × 3)  color crop
    ↓  prepare_blob()
(1 × 3 × 224 × 224)  normalized tensor
    ↓  run_inference()
(1 × 1000)  confidence scores
    ↓  get_top_prediction()
"golden retriever"  94.3%
    ↓  draw_prediction()
annotated image on screen
```

---

## Some ImageNet Categories to Try

| Animals | Vehicles | Objects | Food |
|---------|----------|---------|------|
| golden retriever | sports car | laptop | pizza |
| tabby cat | school bus | backpack | banana |
| bald eagle | ambulance | rocking chair | ice cream |
| hammerhead shark | mountain bike | sunglasses | coffee mug |

Check `synset_words.txt` for the full list of 1000 valid labels.

---

## Reference Docs

- OpenCV DNN:  https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
- OpenCV all:  https://docs.opencv.org/4.x/
- NumPy:       https://numpy.org/doc/stable/reference/