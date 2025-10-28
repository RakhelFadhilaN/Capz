"""
HYBRID SOLUTION: Use your ORIGINAL model (helmet_detector_hardneg.yml)
with improved detection pipeline for multi-person scenarios
"""

import cv2
import numpy as np
import os

class SmartHelmetDetector:
    """
    Uses your original model with smart region detection
    Based on your diagnostic results:
    - Original model: 66% helmet, 83% no-helmet = 72% overall
    - New model: 76% helmet, 50% no-helmet = 66% overall
    
    Original is BETTER!
    """
    def __init__(self, model_path='helmet_detector_camera.yml'):
        self.win_size = (128, 192)  # Your original size
        self.hog = cv2.HOGDescriptor(self.win_size, (16,16), (8,8), (8,8), 9)
        self.svm = cv2.ml.SVM_load(model_path)
        print(f"✓ Loaded original model: {model_path}")
        
        # Based on your diagnostic: optimal threshold around 0.0
        self.helmet_threshold = 0.0
        
        # Region detection parameters
        self.min_person_area = 1500
        self.max_person_area = 80000
    
    def extract_hog_features(self, image):
        """Extract HOG features (original method)"""
        if image.shape[:2] != (self.win_size[1], self.win_size[0]):
            image = cv2.resize(image, self.win_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = self.hog.compute(gray)
        return features.flatten().reshape(1, -1).astype(np.float32)
    
    def find_person_regions_simple(self, image):
        """
        Simple person detection using color segmentation + contours
        More reliable than edge detection for your use case
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for better skin/object detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for potential person regions (skin tones + common colors)
        # This is NOT perfect but works for upper body detection
        lower1 = np.array([0, 20, 70])
        upper1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Also detect darker regions (clothes, hair)
        lower2 = np.array([0, 0, 0])
        upper2 = np.array([180, 255, 100])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_person_area < area < self.max_person_area:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Must be in upper half of image (people are usually visible from top)
                if y < h * 0.7:
                    # Aspect ratio check (people are taller than wide)
                    aspect = h_box / w_box if w_box > 0 else 0
                    if 1.2 < aspect < 5.0:
                        regions.append({
                            'bbox': (x, y, w_box, h_box),
                            'area': area
                        })
        
        # If no regions found, use full image
        if len(regions) == 0:
            regions.append({
                'bbox': (0, 0, w, h),
                'area': w * h
            })
        
        return regions
    
    def classify_region(self, image, bbox):
        """Classify a region using original model"""
        x, y, w, h = bbox
        
        # Add padding
        pad = 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0, 0.0
        
        # Extract features and predict
        features = self.extract_hog_features(region)
        _, prediction = self.svm.predict(features)
        _, decision = self.svm.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        
        pred_class = int(prediction[0][0])
        raw_score = decision[0][0]
        
        # Apply threshold (from your diagnostic: ~0.0 is optimal)
        if raw_score < self.helmet_threshold:
            pred_class = 1  # Helmet
        else:
            pred_class = 0  # No helmet
        
        return pred_class, abs(raw_score)
    
    def detect_helmets(self, image):
        """
        Main detection:
        1. Find person-like regions
        2. Classify each region
        3. Return helmet detections
        """
        regions = self.find_person_regions_simple(image)
        
        detections = []
        for region in regions:
            bbox = region['bbox']
            prediction, confidence = self.classify_region(image, bbox)
            
            detections.append({
                'bbox': bbox,
                'prediction': prediction,
                'score': confidence,
                'label': 'helmet' if prediction == 1 else 'no_helmet'
            })
        
        # Return only helmet detections
        helmet_detections = [d for d in detections if d['prediction'] == 1]
        return helmet_detections, detections  # Return both for debugging
    
    def count_helmets(self, image):
        """Count helmets in image"""
        helmet_detections, all_detections = self.detect_helmets(image)
        return len(helmet_detections), helmet_detections, all_detections
    
    def visualize(self, image, helmet_detections, all_detections=None, show_all=False):
        """Draw results on image"""
        result = image.copy()
        
        # Draw all detected regions (optional)
        if show_all and all_detections:
            for det in all_detections:
                x, y, w, h = det['bbox']
                if det['prediction'] == 0:  # No helmet
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    label = f"No Helmet {det['score']:.2f}"
                    cv2.putText(result, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw helmet detections
        for det in helmet_detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f"HELMET {det['score']:.2f}"
            cv2.putText(result, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Count overlay
        count_text = f"Helmets: {len(helmet_detections)}"
        cv2.rectangle(result, (5, 5), (250, 50), (0, 0, 0), -1)
        cv2.putText(result, count_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return result
    
    def process_image(self, image_path, show_result=True, save_result=False, show_all_regions=False):
        """Process single image"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        img = cv2.imread(image_path)
        if img is None:
            print("✗ Error reading image")
            return None
        
        # Detect
        count, helmet_dets, all_dets = self.count_helmets(img)
        
        print(f"\n✓ Found {len(all_dets)} person region(s)")
        print(f"✓ Detected {count} helmet(s)\n")
        
        for i, det in enumerate(all_dets, 1):
            label = "HELMET" if det['prediction'] == 1 else "NO HELMET"
            print(f"  Person {i}: {label} (confidence: {det['score']:.2f})")
        
        # Visualize
        result = self.visualize(img, helmet_dets, all_dets, show_all=show_all_regions)
        
        if save_result:
            base, ext = os.path.splitext(image_path)
            save_path = f"{base}_result{ext}"
            cv2.imwrite(save_path, result)
            print(f"\n✓ Saved to: {save_path}")
        
        if show_result:
            cv2.imshow('Helmet Detection', result)
            print("\nPress any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return count, helmet_dets
    
    def batch_process(self, folder_path, save_results=False):
        """Process folder"""
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING: {folder_path}")
        print(f"{'='*70}\n")
        
        results = []
        files = sorted([f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        
        for i, filename in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {filename}")
            img_path = os.path.join(folder_path, filename)
            
            try:
                count, dets = self.process_image(
                    img_path, show_result=False, save_result=save_results
                )
                results.append({
                    'filename': filename,
                    'helmet_count': count
                })
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'filename': filename,
                    'helmet_count': 0
                })
        
        # Summary
        print(f"\n{'='*70}")
        print("BATCH SUMMARY")
        print(f"{'='*70}")
        print(f"{'Filename':<40} {'Helmets':<10}")
        print("-"*70)
        
        for r in results:
            print(f"{r['filename']:<40} {r['helmet_count']:<10}")
        
        print(f"{'='*70}\n")
        return results
    
    def adjust_threshold(self, new_threshold):
        """Adjust detection threshold"""
        old = self.helmet_threshold
        self.helmet_threshold = new_threshold
        print(f"✓ Threshold adjusted: {old:.2f} → {new_threshold:.2f}")
        print("  Lower = more helmets detected (more false positives)")
        print("  Higher = fewer helmets detected (more false negatives)")

def main():
    print("="*70)
    print("SMART HELMET DETECTOR - Using Original Model")
    print("="*70)
    
    try:
        detector = SmartHelmetDetector('helmet_detector_camera.yml')
    except:
        print("\n✗ Could not load model!")
        print("Make sure 'helmet_detector_hardneg.yml' exists")
        return
    
    print("\nBased on diagnostic analysis:")
    print(f"  Current threshold: {detector.helmet_threshold:.2f}")
    print("  This gives ~72% overall accuracy on your test set")
    print()
    
    while True:
        print("\n" + "="*70)
        print("MENU")
        print("="*70)
        print("1. Test single image")
        print("2. Batch process folder")
        print("3. Adjust threshold")
        print("4. Test with all regions shown (debug mode)")
        print("5. Exit")
        print("="*70)
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            path = input("Image path: ").strip()
            save = input("Save result? (y/n): ").lower() == 'y'
            detector.process_image(path, show_result=True, save_result=save)
        
        elif choice == '2':
            path = input("Folder path: ").strip()
            save = input("Save results? (y/n): ").lower() == 'y'
            detector.batch_process(path, save_results=save)
        
        elif choice == '3':
            print(f"\nCurrent: {detector.helmet_threshold:.2f}")
            print("Suggested values:")
            print("  -0.2 = Very sensitive (detects more helmets)")
            print("   0.0 = Balanced (recommended)")
            print("   0.2 = Conservative (fewer false positives)")
            new = float(input("New threshold: ").strip())
            detector.adjust_threshold(new)
        
        elif choice == '4':
            path = input("Image path: ").strip()
            save = input("Save result? (y/n): ").lower() == 'y'
            detector.process_image(path, show_result=True, save_result=save, show_all_regions=True)
        
        elif choice == '5':
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()