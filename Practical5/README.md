# TensorFlow.js Image Classification Lab

## Setup (VS Code + Live Server)

1. Open the `image-classification-project` folder in VS Code
2. Install the **Live Server** extension (ritwickdey.liveserver)
3. Right-click `index.html` → **Open with Live Server**
4. Navigate between assignments via the dashboard cards

## Structure

```
image-classification-project/
├── index.html               ← Dashboard homepage
├── assignment1/
│   ├── index.html           ← MobileNet classification UI
│   ├── train.js             ← Classification logic
│   └── images/              ← Sample images (replace with real photos!)
│       ├── dog.jpg
│       ├── car.jpg
│       ├── chair.jpg
│       ├── laptop.jpg
│       └── pineapple.jpg
├── assignment2/
│   ├── index.html           ← Webcam transfer learning UI
│   └── webcam.js            ← Training + prediction logic
└── assignment3/
    ├── index.html           ← Model comparison UI
    └── compare.js           ← Dual-model inference logic
```

## Notes

- **Replace placeholder images** in `assignment1/images/` with real JPG photos for accurate MobileNet predictions
- Assignment 2 requires **webcam access** — allow camera permission in browser
- All models load from CDN — internet connection required on first load
- WebGL backend is used automatically for GPU acceleration
