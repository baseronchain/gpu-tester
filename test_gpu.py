"""
Test GPU Detection untuk CUDA & PyTorch
"""

print("="*60)
print("🔍 GPU DETECTION TEST")
print("="*60)

# Test 1: Check PyTorch
print("\n1️⃣ Testing PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU count: {torch.cuda.device_count()}")
        print(f"   ✅ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ❌ CUDA NOT AVAILABLE - GPU tidak terdeteksi!")
        print("   → Perlu install PyTorch dengan CUDA support")
except ImportError:
    print("   ❌ PyTorch not installed!")

# Test 2: Check Ultralytics
print("\n2️⃣ Testing Ultralytics...")
try:
    from ultralytics import YOLO
    print("   ✅ Ultralytics installed")
    
    # Test model load
    print("   Testing model load...")
    model = YOLO('yolov8n.pt')
    print(f"   ✅ Model loaded on device: {model.device}")
    
except ImportError:
    print("   ❌ Ultralytics not installed!")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# Test 3: Check OpenCV
print("\n3️⃣ Testing OpenCV...")
try:
    import cv2
    print(f"   ✅ OpenCV version: {cv2.__version__}")
    
    # Check CUDA support in OpenCV
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"   ✅ OpenCV built with CUDA support")
    else:
        print(f"   ⚠️ OpenCV without CUDA (not critical)")
except:
    print("   ⚠️ Cannot check OpenCV CUDA support")

# Test 4: Simple inference test
print("\n4️⃣ Testing GPU Inference Speed...")
try:
    import torch
    import time
    
    if torch.cuda.is_available():
        # Test tensor operation on GPU
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        
        start = time.time()
        for _ in range(100):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"   ✅ GPU inference test: {gpu_time:.3f}s")
        print(f"   ✅ GPU is WORKING!")
    else:
        print("   ❌ Cannot test - GPU not available")
        
except Exception as e:
    print(f"   ⚠️ Test failed: {e}")

print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)

# Final verdict
try:
    import torch
    if torch.cuda.is_available():
        print("✅ GPU READY - Anda bisa pakai GPU untuk YOLO!")
        print(f"✅ Device: {torch.cuda.get_device_name(0)}")
        print("\n💡 NEXT STEP:")
        print("   → Gunakan program dengan device='cuda'")
        print("   → Expected FPS dengan RTX 3050: 40-60 FPS (YOLOv8l)")
    else:
        print("❌ GPU NOT AVAILABLE")
        print("\n⚠️ PROBLEM: PyTorch tanpa CUDA support")
        print("\n💡 SOLUTION:")
        print("   → Uninstall PyTorch: pip uninstall torch torchvision")
        print("   → Install PyTorch dengan CUDA:")
        print("   → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
except:
    print("❌ PyTorch not installed")
    print("\n💡 SOLUTION:")
    print("   → Install PyTorch dengan CUDA:")
    print("   → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("="*60)
