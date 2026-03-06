# CV Workshop — Image Classifier

**OpenCV + GoogLeNet (ImageNet)  ·  Beginner Computer Vision Workshop**

---

## Getting Started

First, you will need to fork this repo, ensuring your fork is public

Then, work on the code in this repo, try to get as much done as you can!

Finally, answer the questions listed at the bottom of the README to be entered into the raffle.

And most importantly, have fun!

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
## Questions - MUST BE DONE TO ENTER RAFFLE

1. In your own words, explain why we preprocess the image with grayscale, blur, and edge detection before passing it to the model. What would happen if we skipped one of those steps?
   - Grayscale simplifies the image to intensity only, which makes structural features easier to detect and reduces noise from color variation. Blur smooths small pixel-level noise so edge detection is less jittery. Canny edge detection then highlights object boundaries so we can reliably find a contour and crop the subject ROI before classification.
   - If we skip grayscale, contour extraction becomes less stable because we do not normalize the color channels first. If we skip blur, Canny responds to noise and textures, producing messy edges and bad contours. If we skip edge detection, we lose the boundary map used by `find_subject_contour()`, so ROI selection can fail or include too much background, hurting classification.

2. When you ran your classifier on an image, what did it predict and how confident was it? Did the result surprise you — and if it got something wrong, why do you think that happened?
   - From `python3 main.py --batch images/`, `car.jpg` was predicted as **car wheel** with **86.1%** confidence (best result), which is plausible because the crop likely focused on a wheel-like region.
   - I also saw errors/surprises: `cat.jpg` predicted **paintbrush** at 45.1% and `dog.jpg` predicted **miniature pinscher** at 20.9% (low confidence). Likely causes: imperfect contour-based cropping, limited ImageNet label alignment with our expected labels, and background/context dominating the ROI.

3. We focused on the top prediction (the supposed classification) — but the model outputs 1000 scores simultaneously. What does it mean that the scores for other classes are non-zero? What are those numbers telling you?
   - The model is producing a probability distribution over all 1000 ImageNet classes. Non-zero scores mean the model sees partial evidence for multiple classes, not just one.
   - Higher non-top scores often indicate semantic similarity (for example, visually related dog breeds), shared textures/shapes, or uncertainty from ambiguous input. The gap between top-1 and runner-up is a useful confidence signal: a large gap usually means a clearer decision; a small gap means the model is uncertain.

4. Where would you take this project next? Think about different models you could swap in, new kinds of images you'd want to classify, or features you'd add to make it more useful in the real world.
   - Swap in stronger modern backbones (for example EfficientNet/ConvNeXt or a lightweight MobileNet variant) and compare top-1/top-5 accuracy plus latency.
   - Improve localization by replacing contour heuristics with an object detector (YOLO/SSD) so classification uses cleaner subject crops.
   - Add top-k visualization, confidence calibration, and reject/abstain logic for low-confidence predictions.
   - Build a simple API or webcam app for real-time inference, plus logging/metrics to track failures and guide future fine-tuning on domain-specific data.

## Reference Docs

- OpenCV DNN:  https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
- OpenCV all:  https://docs.opencv.org/4.x/
- NumPy:       https://numpy.org/doc/stable/reference/
