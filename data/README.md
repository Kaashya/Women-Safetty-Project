# Dataset Setup & Instructions

This directory holds local video clips used for testing and validating the **Women Safety Analytics** prototype.

> **IMPORTANT: DATASET VIDEOS MUST NOT BE COMMITTED TO GIT.**  
> Raw `.mp4`, `.avi`, `.mov`, and other video formats are explicitly ignored by `.gitignore`. Keep only `.gitkeep` tracked.

---

## 1. Dataset Source

We primarily test with the **Le2i Fall Detection Dataset** (or any CCTV-style surveillance videos featuring normal walking/sitting and fall scenarios):
- **Le2i Fall Detection Dataset**: Contains realistic surveillance scenarios (Home, Coffee room, Office, Lecture room) captured with a single fixed camera.
- Alternatively, public CCTV footage clips or recorded webcam clips can be used.

---

## 2. Recommended Directory Organization

Download your representative test clips and organize them into `data/normal/` and `data/fall/`:

```text
data/
├── README.md
├── normal/
│   ├── .gitkeep
│   ├── walking_01.mp4
│   ├── walking_02.mp4
│   ├── sitting_01.mp4
│   └── standing_01.mp4
│
└── fall/
    ├── .gitkeep
    ├── fall_01.mp4
    ├── fall_02.mp4
    ├── fall_03.mp4
    └── fall_04.mp4
```

---

## 3. Prototype Scope Note

Because we use **pretrained YOLO** for person detection and tracking, **no model training is performed**. Therefore, you do **NOT** need hundreds of gigabytes of training videos. A small representative set of 3–5 normal clips and 3–5 fall clips is sufficient for prototype verification and demonstration across all 3 members.
