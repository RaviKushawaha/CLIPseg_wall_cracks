import os, argparse
from huggingface_hub import HfApi, login
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


def load_model(model_id):
    """
    Load CLIPSeg model and processor from Hugging Face.
    
    Args:
        model_id: HuggingFace model ID (e.g., "your-username/model-name")
    
    Returns:
        model, processor, device
    """
    print(f"Loading model: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = CLIPSegProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        ignore_mismatched_sizes=True
    )
    model = CLIPSegForImageSegmentation.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        ignore_mismatched_sizes=True
    )
    
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")
    
    return model, processor, device


def save_segmentation_mask(image_path, text_prompt, model, processor, device, threshold=0.5):
    """
    Segment an image using text prompt and display result.
    
    Args:
        image_path: Path to image file or PIL Image object
        text_prompt: Text description of what to segment (e.g., "a dog")
        model: CLIPSeg model
        processor: CLIPSeg processor
        device: torch device (cuda/cpu)
        threshold: Confidence threshold for segmentation (0-1)
    
    Returns:
        mask: Binary segmentation mask (numpy array)
    """
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
        
    # Prepare inputs
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Resize to original image size
        original_size = image.size[::-1]  # (height, width)
        logits = F.interpolate(
            logits.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False
        ).squeeze()
        
        # Get binary mask
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float().cpu().numpy()
        binary_mask = (mask * 255).astype('uint8')

        output_path = f"{os.path.basename(image_path)}__{text_prompt.replace(' ','_')}.png"
        Image.fromarray(binary_mask).save(output_path)
        

def hugging_face_login(hf_token):
    login(token=hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment image using text prompt')
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text prompt for segmentation')
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face token')
    parser.add_argument('--model_id', type=str, default='RaviKush/clipseg_focal_loss_v1', help='HuggingFace model ID')
    args = parser.parse_args()

    hugging_face_login(args.hf_token)
    model, processor, device = load_model(args.model_id)
    save_segmentation_mask(args.img_path, args.text_prompt, model, processor, device)