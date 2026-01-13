"""
YOLO-SCEMA Model Export Script

Export YOLO-SCEMA model to various formats including:
- TorchScript
- ONNX
- TensorRT
- CoreML
- TFLite

Author: Mohammed MAIZA
Repository: https://github.com/MAIZA-MOHAMMED/Spatial-Channel-Enhanced-Multiscale-Attention
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models import create_model

class ModelExporter:
    """YOLO-SCEMA Model Exporter"""
    
    def __init__(self, config_path, weights_path):
        """
        Initialize exporter
        
        Args:
            config_path: Path to configuration file
            weights_path: Path to model weights
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Setup device
        self.device = torch.device('cpu')  # Export always on CPU
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(weights_path)
        
        # Export configuration
        self.export_config = self.config.get('export', {})
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_model(self, weights_path):
        """Load YOLO-SCEMA model"""
        model_config = self.config['model']
        
        # Get model size
        model_size = model_config.get('size', 'n')
        
        # Get number of classes
        if 'head' in model_config and 'num_classes' in model_config['head']:
            num_classes = model_config['head']['num_classes']
        else:
            # Get from dataset config
            dataset_name = self.config['data'].get('dataset', 'coco')
            num_classes = self.config['datasets'][dataset_name]['num_classes']
        
        # Create model
        model = create_model(
            model_size=model_size,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        if weights_path is None:
            raise ValueError("Weights path must be provided")
        
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"Loaded weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Move to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        # Print model info
        print(f"Model: YOLO-SCEMA-{model_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Input size: {self.config['data']['img_size']}")
        
        return model
    
    def export_torchscript(self, output_path=None):
        """
        Export model to TorchScript
        
        Args:
            output_path: Output path for TorchScript model
        """
        print("\n" + "="*50)
        print("Exporting to TorchScript")
        print("="*50)
        
        if output_path is None:
            output_path = f"yolo_scema_{self.config['model'].get('size', 'n')}.torchscript.pt"
        
        # Create example input
        img_size = self.config['data']['img_size']
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        example_input = torch.randn(1, 3, img_size[0], img_size[1], device=self.device)
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Save traced model
            torch.jit.save(traced_model, output_path)
            
            # Verify the exported model
            loaded_model = torch.jit.load(output_path)
            with torch.no_grad():
                output = loaded_model(example_input)
            
            print(f"✓ TorchScript model exported successfully")
            print(f"  Output path: {output_path}")
            print(f"  Input shape: {example_input.shape}")
            print(f"  Output shapes: {[o.shape for o in output]}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to export TorchScript model: {e}")
            return False
    
    def export_onnx(self, output_path=None, dynamic=False):
        """
        Export model to ONNX
        
        Args:
            output_path: Output path for ONNX model
            dynamic: Whether to enable dynamic axes
        """
        print("\n" + "="*50)
        print("Exporting to ONNX")
        print("="*50)
        
        if output_path is None:
            output_path = f"yolo_scema_{self.config['model'].get('size', 'n')}.onnx"
        
        # Get ONNX configuration
        onnx_config = self.export_config.get('onnx', {})
        opset_version = onnx_config.get('opset_version', 12)
        simplify = onnx_config.get('simplify', True)
        
        # Create example input
        img_size = self.config['data']['img_size']
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        example_input = torch.randn(1, 3, img_size[0], img_size[1], device=self.device)
        
        # Define dynamic axes if requested
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            print(f"✓ ONNX model exported successfully")
            print(f"  Output path: {output_path}")
            print(f"  Opset version: {opset_version}")
            print(f"  Dynamic axes: {dynamic}")
            
            # Simplify ONNX model if requested
            if simplify:
                try:
                    import onnx
                    from onnxsim import simplify
                    
                    # Load ONNX model
                    onnx_model = onnx.load(output_path)
                    
                    # Simplify
                    model_simp, check = simplify(onnx_model)
                    
                    if check:
                        # Save simplified model
                        simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                        onnx.save(model_simp, simplified_path)
                        print(f"  Simplified model saved to: {simplified_path}")
                    else:
                        print("  Warning: Simplified model check failed")
                        
                except ImportError:
                    print("  Warning: onnx-simplifier not installed, skipping simplification")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to export ONNX model: {e}")
            return False
    
    def export_tensorrt(self, output_path=None, fp16=True):
        """
        Export model to TensorRT
        
        Args:
            output_path: Output path for TensorRT engine
            fp16: Whether to use FP16 precision
        """
        print("\n" + "="*50)
        print("Exporting to TensorRT")
        print("="*50)
        
        try:
            import tensorrt as trt
            
            if output_path is None:
                output_path = f"yolo_scema_{self.config['model'].get('size', 'n')}.trt"
            
            # Get TensorRT configuration
            trt_config = self.export_config.get('tensorrt', {})
            workspace_size = trt_config.get('workspace_size', 4) * 1024 * 1024 * 1024  # Convert to bytes
            max_batch_size = trt_config.get('max_batch_size', 32)
            
            # First export to ONNX
            onnx_path = output_path.replace('.trt', '.onnx')
            if not self.export_onnx(onnx_path, dynamic=False):
                return False
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            
            # Build configuration
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("  Using FP16 precision")
            
            # Build engine
            engine = builder.build_serialized_network(network, config)
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine)
            
            print(f"✓ TensorRT engine exported successfully")
            print(f"  Output path: {output_path}")
            print(f"  Workspace size: {workspace_size / (1024**3):.1f} GB")
            print(f"  Max batch size: {max_batch_size}")
            print(f"  FP16 enabled: {fp16}")
            
            # Clean up temporary ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            
            return True
            
        except ImportError:
            print("✗ TensorRT not installed. Install with: pip install tensorrt")
            return False
        except Exception as e:
            print(f"✗ Failed to export TensorRT engine: {e}")
            return False
    
    def export_coreml(self, output_path=None):
        """
        Export model to CoreML
        
        Args:
            output_path: Output path for CoreML model
        """
        print("\n" + "="*50)
        print("Exporting to CoreML")
        print("="*50)
        
        try:
            import coremltools as ct
            
            if output_path is None:
                output_path = f"yolo_scema_{self.config['model'].get('size', 'n')}.mlpackage"
            
            # Get CoreML configuration
            coreml_config = self.export_config.get('coreml', {})
            compute_units = coreml_config.get('compute_units', 'ALL')
            min_deployment_target = coreml_config.get('minimum_deployment_target', 'iOS13')
            
            # First export to ONNX
            onnx_path = output_path.replace('.mlpackage', '.onnx')
            if not self.export_onnx(onnx_path, dynamic=False):
                return False
            
            # Convert ONNX to CoreML
            model = ct.convert(
                onnx_path,
                convert_to=min_deployment_target,
                compute_units=getattr(ct.ComputeUnit, compute_units)
            )
            
            # Save CoreML model
            model.save(output_path)
            
            print(f"✓ CoreML model exported successfully")
            print(f"  Output path: {output_path}")
            print(f"  Minimum deployment target: {min_deployment_target}")
            print(f"  Compute units: {compute_units}")
            
            # Clean up temporary ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            
            return True
            
        except ImportError:
            print("✗ CoreML Tools not installed. Install with: pip install coremltools")
            return False
        except Exception as e:
            print(f"✗ Failed to export CoreML model: {e}")
            return False
    
    def export_tflite(self, output_path=None, quantize=False):
        """
        Export model to TFLite
        
        Args:
            output_path: Output path for TFLite model
            quantize: Whether to apply quantization
        """
        print("\n" + "="*50)
        print("Exporting to TFLite")
        print("="*50)
        
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            
            if output_path is None:
                output_path = f"yolo_scema_{self.config['model'].get('size', 'n')}.tflite"
            
            # First export to ONNX
            onnx_path = output_path.replace('.tflite', '.onnx')
            if not self.export_onnx(onnx_path, dynamic=False):
                return False
            
            # Convert ONNX to TensorFlow
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            
            # Export TensorFlow model
            tf_model_path = output_path.replace('.tflite', '_tf')
            tf_rep.export_graph(tf_model_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✓ TFLite model exported successfully")
            print(f"  Output path: {output_path}")
            print(f"  Quantized: {quantize}")
            
            # Clean up temporary files
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            if os.path.exists(tf_model_path):
                import shutil
                shutil.rmtree(tf_model_path)
            
            return True
            
        except ImportError as e:
            print(f"✗ Required packages not installed: {e}")
            print("  Install with: pip install tensorflow onnx onnx-tf")
            return False
        except Exception as e:
            print(f"✗ Failed to export TFLite model: {e}")
            return False
    
    def export_all(self, output_dir='./exports'):
        """
        Export model to all supported formats
        
        Args:
            output_dir: Output directory for exported models
        """
        print("\n" + "="*50)
        print("Exporting to All Formats")
        print("="*50)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model size for naming
        model_size = self.config['model'].get('size', 'n')
        base_name = f"yolo_scema_{model_size}"
        
        # Get export formats from config
        formats = self.export_config.get('formats', ['torchscript', 'onnx'])
        
        results = {}
        
        # Export to each format
        for fmt in formats:
            fmt = fmt.lower()
            
            if fmt == 'torchscript':
                output_path = output_dir / f"{base_name}.torchscript.pt"
                results['torchscript'] = self.export_torchscript(str(output_path))
                
            elif fmt == 'onnx':
                output_path = output_dir / f"{base_name}.onnx"
                dynamic = self.export_config.get('onnx', {}).get('dynamic_axes', True)
                results['onnx'] = self.export_onnx(str(output_path), dynamic=dynamic)
                
            elif fmt == 'tensorrt':
                output_path = output_dir / f"{base_name}.trt"
                fp16 = self.export_config.get('tensorrt', {}).get('fp16', True)
                results['tensorrt'] = self.export_tensorrt(str(output_path), fp16=fp16)
                
            elif fmt == 'coreml':
                output_path = output_dir / f"{base_name}.mlpackage"
                results['coreml'] = self.export_coreml(str(output_path))
                
            elif fmt == 'tflite':
                output_path = output_dir / f"{base_name}.tflite"
                quantize = self.export_config.get('tflite', {}).get('quantize', False)
                results['tflite'] = self.export_tflite(str(output_path), quantize=quantize)
                
            else:
                print(f"Warning: Unknown format '{fmt}', skipping")
        
        # Print summary
        print("\n" + "="*50)
        print("Export Summary")
        print("="*50)
        
        successful = 0
        total = len(results)
        
        for fmt, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{fmt:15s}: {status}")
            if success:
                successful += 1
        
        print(f"\nSuccessful exports: {successful}/{total}")
        print(f"Output directory: {output_dir.absolute()}")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Export YOLO-SCEMA model')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--config', type=str, default='configs/yolov8_SCEMA.yaml',
                       help='Path to configuration file')
    parser.add_argument('--format', type=str, default='all',
                       help='Export format: torchscript, onnx, tensorrt, coreml, tflite, all')
    parser.add_argument('--output', type=str, default='./exports',
                       help='Output directory or file path')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision (for TensorRT)')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization (for TFLite)')
    parser.add_argument('--dynamic', action='store_true',
                       help='Enable dynamic axes (for ONNX)')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ModelExporter(args.config, args.weights)
    
    # Perform export
    export_format = args.format.lower()
    
    if export_format == 'all':
        exporter.export_all(args.output)
    
    elif export_format == 'torchscript':
        output_path = args.output if args.output.endswith('.pt') else None
        exporter.export_torchscript(output_path)
    
    elif export_format == 'onnx':
        output_path = args.output if args.output.endswith('.onnx') else None
        exporter.export_onnx(output_path, dynamic=args.dynamic)
    
    elif export_format == 'tensorrt':
        output_path = args.output if args.output.endswith('.trt') else None
        exporter.export_tensorrt(output_path, fp16=args.fp16)
    
    elif export_format == 'coreml':
        output_path = args.output if args.output.endswith('.mlpackage') else None
        exporter.export_coreml(output_path)
    
    elif export_format == 'tflite':
        output_path = args.output if args.output.endswith('.tflite') else None
        exporter.export_tflite(output_path, quantize=args.quantize)
    
    else:
        print(f"Error: Unknown format '{export_format}'")
        print("Supported formats: torchscript, onnx, tensorrt, coreml, tflite, all")

if __name__ == '__main__':
    main()
