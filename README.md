# CLIPSeg Image Segmentation - Inference

This script performs text-guided image segmentation using a fine-tuned CLIPSeg model.

## 🎓 Model Training

- Dataset was obtained from two sources:
  - https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36
  - https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect
- CLIPSeg model was trained in two stages: first only decoder, then full model
- `Kaggle_notebooks` folder contains the training notebooks

## 📋 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Usage
Run the script from the command line:

```bash
  python inference.py \
  --img_path "sample.jpg" \
  --text_prompt "segment crack" \
  --hf_token "hf_xxxxxxxxxxxxx" \
  --model_id "RaviKush/clipseg_focal_loss_v1"

```


## 📤 Output

The script will:
- Load the model from Hugging Face
- Process given image with the text prompt
- Save a segmentation mask as: `{image_name}__{text_prompt}.png`

**Example output filename:** `123__segment_crack.png.`

The mask is a PNG image with:
- **Black pixels (0)**: Background
- **White pixels (255)**: Segmented object

## 📝 Notes

* Inference works seamlessly on both CPU and GPU.
* Detailed analysis provided in report.pdf

