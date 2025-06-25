import cv2
import time
import easyocr
from rapidocr_onnxruntime import RapidOCR
from fast_plate_ocr import ONNXPlateRecognizer

img_path = "./plate_crops/plate_1_0_0.62.jpg"
img = cv2.imread(img_path)

sharpened = cv2.addWeighted(img, 1.1, img, 0, 0)
enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 15)

cv2.imshow("Sharpened", sharpened)
cv2.waitKey(0)
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)
cv2.imshow("Denoised", denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale for ONNXPlateRecognizer
gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

# Test EasyOCR
print("Testing EasyOCR...")
reader_easy = easyocr.Reader(['en'])
start_time = time.time()
easy_result = reader_easy.readtext(denoised, detail=0)
easy_time = time.time() - start_time
print(f"EasyOCR result: {easy_result}")
print(f"EasyOCR time: {easy_time:.4f} seconds")

# Test RapidOCR
print("\nTesting RapidOCR...")
reader_rapid = RapidOCR()
start_time = time.time()
rapid_result, _ = reader_rapid(denoised, return_img=True)
rapid_time = time.time() - start_time
print(f"RapidOCR result: {[text[1] for text in rapid_result]}")
print(f"RapidOCR time: {rapid_time:.4f} seconds")

# Test ONNXPlateRecognizer
print("\nTesting ONNXPlateRecognizer...")
m = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model', providers=['CPUExecutionProvider'])
start_time = time.time()
onnx_result = m.run(gray_denoised)
onnx_time = time.time() - start_time
print(f"ONNXPlateRecognizer result: {onnx_result}")
print(f"ONNXPlateRecognizer time: {onnx_time:.4f} seconds")

# Determine the fastest model
print("\n--- Performance Comparison ---")
times = {
    "EasyOCR": easy_time,
    "RapidOCR": rapid_time,
    "ONNXPlateRecognizer": onnx_time
}

fastest_model = min(times, key=times.get)
print(f"Fastest model: {fastest_model} ({times[fastest_model]:.4f} seconds)")
print("All times:")
for model, t in sorted(times.items(), key=lambda x: x[1]):
    print(f"  {model}: {t:.4f} seconds")