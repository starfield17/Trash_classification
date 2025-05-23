import os
import torch
import torchvision
import gc
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Class names - keep consistent with training script
CLASS_NAMES = ["Kitchen waste", "Recyclable waste", "Hazardous waste", "Other waste"]

# Select the model type to use, ensure it matches the training configuration
MODEL_TYPE = "resnet50_fpn"  # Options: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3", "resnet50_fpn_v2"


def get_faster_rcnn_model(num_classes, model_type="resnet50_fpn"):
    """Get different types of Faster R-CNN models"""
    # num_classes needs +1 because 0 is the background class
    num_classes_with_bg = num_classes + 1
    
    if model_type == "resnet50_fpn":
        # Standard version: ResNet50+FPN
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    elif model_type == "resnet18_fpn":
        # Lightweight version: ResNet18+FPN
        backbone = resnet_fpn_backbone(
            'resnet18', 
            pretrained=True, 
            trainable_layers=3
        )
        
        # Set up anchor generator for RPN
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        # Set up RoI pooling size
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create Faster R-CNN model
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes_with_bg,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    elif model_type == "mobilenet_v3":
        # Ultra-lightweight version: MobileNetV3
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    elif model_type == "_v2":
        model = fasterrcnn__v2(weights=FasterRCNN__V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def save_optimized_model(model, output_dir, device, model_type):
    """Save the optimized model, including different formats and precisions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save basic model
    model_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Export model (TorchScript format)
    try:
        # Switch to evaluation mode
        model.eval()
        
        # Create scripted model with example input
        dummy_input = [torch.rand(3, 640, 640).to(device)]
        script_model = torch.jit.trace(model, dummy_input)
        script_model_path = os.path.join(output_dir, "model_scripted.pt")
        torch.jit.save(script_model, script_model_path)
        print(f"TorchScript model saved to: {script_model_path}")
    except Exception as e:
        print(f"TorchScript export failed: {e}")
    
    # If CUDA is available, save half-precision model
    if device != "cpu" and torch.cuda.is_available():
        try:
            model_fp16 = model.half()
            fp16_path = os.path.join(output_dir, "model_fp16.pth")
            torch.save(model_fp16.state_dict(), fp16_path)
            print(f"FP16 model saved to: {fp16_path}")
            
            # Restore to FP32
            model = model.float()
        except Exception as e:
            print(f"FP16 model save failed: {e}")
    
    # Export ONNX model (for Raspberry Pi deployment)
    try:
        dummy_input = torch.rand(1, 3, 640, 640).to(device)
        input_names = ["input"]
        output_names = ["boxes", "labels", "scores"]
        
        # Temporarily create a forward function that supports ONNX export
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, x):
                predictions = self.model([x])
                return (
                    predictions[0]["boxes"], 
                    predictions[0]["labels"], 
                    predictions[0]["scores"]
                )
        
        wrapper = ModelWrapper(model)
        
        onnx_path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(
            wrapper, 
            dummy_input, 
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11
        )
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    # Model quantization (INT8) - better for running on Raspberry Pi
    try:
        # Note: Actual quantization usually requires a calibration dataset
        # Here we just show a simplified quantization process
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        q_path = os.path.join(output_dir, "model_quantized.pth")
        torch.save(quantized_model.state_dict(), q_path)
        print(f"Quantized model saved to: {q_path}")
    except Exception as e:
        print(f"Quantized model save failed: {e}")
    
    print("\nModel export completed!")
    return model_path


def convert_checkpoint_to_final_model(checkpoint_path, output_dir, model_type=MODEL_TYPE):
    """Convert specified checkpoint file to final model"""
    try:
        print(f"Starting conversion of checkpoint {checkpoint_path} to final model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create new model
        print(f"Creating model architecture ({model_type})...")
        model = get_faster_rcnn_model(len(CLASS_NAMES), model_type=model_type)
        
        # Load model parameters from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("Successfully loaded model parameters")
        
        # Switch to evaluation mode
        model.eval()
        
        # Save optimized model
        print("\nSaving and optimizing model for deployment...")
        model_path = save_optimized_model(model, output_dir, device, model_type)
        
        print(f"\nConversion completed! Model saved in {output_dir} directory")
        print(f"Final model path: {model_path}")
        print("You can use the saved ONNX or TorchScript model for deployment on Raspberry Pi.")
        
        return model_path
    
    except Exception as e:
        print(f"\nError occurred during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("\nScript execution completed. Cleaning up...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Convert training checkpoint model to deployable final model')
    parser.add_argument('--model_path', type=str, default='./output/model_epoch_35.pth',
                        help='Checkpoint file path (default: ./output/model_epoch_35.pth)')
    parser.add_argument('--out_dir', type=str, default='./output/final_model',
                        help='Output directory (default: ./output/final_model)')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE,
                        choices=['resnet50_fpn', 'resnet18_fpn', 'mobilenet_v3', 'resnet50_fpn_v2'],
                        help=f'Model type (default: {MODEL_TYPE})')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    print(f"Using parameters:")
    print(f"  - Model path: {args.model_path}")
    print(f"  - Output directory: {args.out_dir}")
    print(f"  - Model type: {args.model_type}")
    
    # Execute conversion
    convert_checkpoint_to_final_model(args.model_path, args.out_dir, args.model_type)
