import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import subprocess
import sys
import os
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# Global control variables
DEBUG_WINDOW = False
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9  # Confidence threshold
model_path = "yolov12n_e300.pt"

# Serial port configuration
# Available serial port mapping (raspberrypi):
# Port Name | TX Pin  | RX Pin
# ttyS0    | GPIO14 | GPIO15
# ttyAMA2  | GPIO0  | GPIO1
# ttyAMA3  | GPIO4  | GPIO5
# ttyAMA4  | GPIO8  | GPIO9
# ttyAMA5  | GPIO12 | GPIO13

STM32_PORT = "/dev/ttyUSB0"  # choose serial if you want
STM32_BAUD = 115200
CAMERA_WIDTH = 1280  # Camera width
CAMERA_HEIGHT = 720  # Camera height
MAX_SERIAL_VALUE = 255  # Maximum value for serial transmission


class SerialManager:
    def __init__(self):
        self.stm32_port = None
        self.is_running = True
        self.last_stm32_send_time = 0
        self.MIN_SEND_INTERVAL = 0.1  # Minimum sending interval (seconds)

        # Current frame detection results
        self.current_detections = []

        # Serial sending queue related
        self.send_queue = []
        self.queue_lock = threading.Lock()
        self.MAX_QUEUE_SIZE = 50  # Maximum queue length

        # Garbage counting and recording related
        self.garbage_count = 0
        self.detected_items = []

        # Anti-duplicate counting and stability detection related
        self.last_count_time = 0
        self.COUNT_COOLDOWN = 5.0
        self.is_counting_locked = False
        self.last_detected_type = None

        # Stability detection related
        self.current_detection = None
        self.detection_start_time = 0
        self.STABILITY_THRESHOLD = 1.0
        self.stable_detection = False
        self.detection_lost_time = 0
        self.DETECTION_RESET_TIME = 0.5

        # Classification mapping
        waste_classifier = WasteClassifier()
        self.zero_mapping = max(waste_classifier.class_names.keys()) + 1
        print(f"Class 0 will be mapped to: {self.zero_mapping}")

        # Initialize STM32 serial port
        if ENABLE_SERIAL:
            try:
                self.stm32_port = serial.Serial(
                    STM32_PORT, STM32_BAUD, timeout=0.1, write_timeout=0.1
                )
                print(f"STM32 serial port initialized: {STM32_PORT}")

                # Start queue processing thread
                self.queue_thread = threading.Thread(target=self.queue_processor_thread)
                self.queue_thread.daemon = True
                self.queue_thread.start()
                print("Serial sending queue processing thread started")
            except Exception as e:
                print(f"STM32 serial port initialization failed: {str(e)}")
                self.stm32_port = None

    def check_detection_stability(self, garbage_type):
        """Check detection stability"""
        current_time = time.time()

        # If a new object type is detected, or detection interruption exceeds reset time
        if garbage_type != self.current_detection or (
            current_time - self.detection_lost_time > self.DETECTION_RESET_TIME
            and self.detection_lost_time > 0
        ):
            # Reset detection state
            self.current_detection = garbage_type
            self.detection_start_time = current_time
            self.stable_detection = False
            self.detection_lost_time = 0
            return False
        
        # If stable recognition time has been reached
        if (
            current_time - self.detection_start_time >= self.STABILITY_THRESHOLD
            and not self.stable_detection
        ):
            self.stable_detection = True
            return True
        return self.stable_detection

    def can_count_new_garbage(self, garbage_type):
        """Check if new garbage can be counted"""
        current_time = time.time()

        # Check stability
        if not self.check_detection_stability(garbage_type):
            return False
        
        # If it's a new garbage type, reset lock state
        if garbage_type != self.last_detected_type:
            self.is_counting_locked = False
            self.last_detected_type = garbage_type
        
        # Check if within cooldown time
        if self.is_counting_locked:
            if current_time - self.last_count_time >= self.COUNT_COOLDOWN:
                self.is_counting_locked = False  # Release lock
            else:
                return False
        return True

    def update_garbage_count(self, garbage_type):
        """Update garbage count"""
        if not self.can_count_new_garbage(garbage_type):
            return
        
        self.garbage_count += 1
        self.detected_items.append(
            {
                "count": self.garbage_count,
                "type": garbage_type,
                "quantity": 1,
                "status": "Correct",
            }
        )

        # Update counting related state
        self.last_count_time = time.time()
        self.is_counting_locked = True
        self.last_detected_type = garbage_type

    def clear_detections(self):
        """Clear current frame detection results"""
        self.current_detections = []

    def add_detection(self, class_id, center_x, center_y):
        """Add new detection result to temporary list"""
        self.current_detections.append((class_id, center_x, center_y))

    def queue_processor_thread(self):
        """Queue processing thread, periodically take data from queue and send"""
        print("Queue processing thread started")
        while self.is_running:
            try:
                self._process_queue_batch()
            except Exception as e:
                print(f"Queue processing exception: {str(e)}")
            # Control processing frequency
            time.sleep(self.MIN_SEND_INTERVAL / 2)

    def _process_queue_batch(self):
        """Process data in sending queue"""
        # 1. Check serial port status
        if not self.stm32_port:
            print("Error: Serial port object does not exist")
            return
        
        if not self.stm32_port.is_open:
            try:
                print("Attempting to reopen serial port...")
                self.stm32_port.open()
                print("Serial port reopened successfully")
            except Exception as e:
                print(f"Serial port reopen failed: {str(e)}")
                return

        # 2. Control sending frequency
        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return  # Not time to send, wait for next processing

        # 3. Get one data from queue
        data_to_send = None
        with self.queue_lock:
            if self.send_queue:
                data_to_send = self.send_queue.pop(0)

        if not data_to_send:
            return

        try:
            # 4. Data preparation
            # Simplified: not resetting buffer, may cause additional delay
            # self.stm32_port.reset_input_buffer()
            # self.stm32_port.reset_output_buffer()
            # time.sleep(0.01)

            # 5. Assemble data packet, add header and footer identifiers
            data = bytes(
                [
                    data_to_send["class_id"],
                    data_to_send["x"],
                    data_to_send["y"],
                ]
            )

            # 6. Send data
            bytes_written = self.stm32_port.write(data)
            self.stm32_port.flush()
            self.last_stm32_send_time = current_time

            if DEBUG_WINDOW:
                print("\n----- Serial Send Detailed Data [DEBUG] -----")
                print(f"Hexadecimal data: {' '.join([f'0x{b:02X}' for b in data])}")

                # Add more intuitive raw data packet display
                print("Raw data packet structure:")
                print(
                    f"  [0] 0x{data[0]:02X} - Class ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})"
                )
                print(
                    f"  [1] 0x{data[1]:02X} - X coordinate ({data_to_send['orig_x']} -> {data_to_send['x']})"
                )
                print(
                    f"  [2] 0x{data[2]:02X} - Y coordinate ({data_to_send['orig_y']} -> {data_to_send['y']})"
                )
                print(f"Total packet length: {len(data)} bytes, actually written: {bytes_written} bytes")
                print(
                    f"Original class ID: {data_to_send['orig_class']} (decimal) -> {data_to_send['class_id']} (send value)"
                )
                print(
                    f"Original coordinates: ({data_to_send['orig_x']}, {data_to_send['orig_y']}) -> Scaled: (0x{data_to_send['x']:02X}, 0x{data_to_send['y']:02X})"
                )
                print(
                    f"Data waiting time in queue: {current_time - data_to_send['timestamp']:.3f} seconds"
                )
                print("-" * 50)

        except serial.SerialTimeoutException:
            print("Serial write timeout, device may not be responding")
            # Put timeout data back to queue to keep data from being lost
            with self.queue_lock:
                self.send_queue.insert(0, data_to_send)
        except Exception as e:
            print(f"Serial sending error: {str(e)}")
            # Consider putting back to queue when sending fails
            with self.queue_lock:
                # Only put back to queue when retry count doesn't exceed limit
                retry_count = data_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # Maximum 3 retries
                    data_to_send["retry"] = retry_count
                    self.send_queue.insert(0, data_to_send)
                    print(f"Data will retry sending, attempt {retry_count}")

    def send_to_stm32(self, class_id, center_x, center_y):
        """Send data to STM32, using queue to ensure reliable data transmission"""
        # Check serial port and queue thread status
        if not self.stm32_port or not self.stm32_port.is_open:
            print("Warning: Serial port not initialized or not open, cannot send data")
            return False

        # Check if queue thread is running
        if hasattr(self, "queue_thread") and not self.queue_thread.is_alive():
            print("Warning: Queue processing thread not running, attempting to restart")
            try:
                self.queue_thread = threading.Thread(target=self.queue_processor_thread)
                self.queue_thread.daemon = True
                self.queue_thread.start()
                print("Serial sending queue processing thread restarted")
            except Exception as e:
                print(f"Failed to start queue thread: {str(e)}")
                return False

        # Ensure data is within valid range
        if class_id == 0:
            mapped_class_id = self.zero_mapping
        else:
            mapped_class_id = class_id

        # Force limit all values within 0-255 range
        mapped_class_id = min(255, max(0, mapped_class_id))
        x_scaled = min(
            MAX_SERIAL_VALUE, max(0, int(center_x * MAX_SERIAL_VALUE / CAMERA_WIDTH))
        )
        y_scaled = min(
            MAX_SERIAL_VALUE, max(0, int(center_y * MAX_SERIAL_VALUE / CAMERA_HEIGHT))
        )

        # Add to sending queue
        with self.queue_lock:
            # Limit queue size to avoid excessive memory usage
            if len(self.send_queue) >= 10:  # Reduce queue size limit to avoid too much backlog
                # Keep latest data, discard old data
                self.send_queue = self.send_queue[-9:]
                print("Warning: Send queue is full, discarding old data")

            # Add data to queue, including original values and scaled values
            self.send_queue.append(
                {
                    "class_id": mapped_class_id,
                    "x": x_scaled,
                    "y": y_scaled,
                    "timestamp": time.time(),
                    "orig_x": center_x,
                    "orig_y": center_y,
                    "orig_class": class_id,
                    "retry": 0,  # Initial retry count
                }
            )
        return True

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        print("Cleaning up resources...")

        # Wait for queue processing thread to end
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("Queue processing thread terminated")

        # Close serial port
        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
            print("Serial port closed")


class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        # Updated to four major categories
        self.class_names = {
            0: "Kitchen Waste",
            1: "Recyclable Waste",
            2: "Hazardous Waste",
            3: "Other Waste",
        }
        # Specify fixed colors for each category
        self.colors = {
            0: (86, 180, 233),  # Kitchen Waste - Blue
            1: (230, 159, 0),   # Recyclable Waste - Orange
            2: (240, 39, 32),   # Hazardous Waste - Red
            3: (0, 158, 115),   # Other Waste - Green
        }
        self.serial_manager = SerialManager()

        # Add detection throttling related variables
        self.last_detection_time = 0
        self.detection_interval = 0.5  # Send data at most once every 0.5 seconds
        self.last_detection_dict = {}  # Record last detection position and time for each category
        self.min_position_change = 20  # Position change threshold (pixels)

    def _should_send_detection(self, class_id, center_x, center_y, confidence):
        """Determine if current detection result should be sent"""
        current_time = time.time()

        # Check confidence
        if confidence < CONF_THRESHOLD:
            return False

        # Check time interval
        if current_time - self.last_detection_time < self.detection_interval:
            # Insufficient time interval, check for significant changes

            # If it's a new category, send it
            if class_id not in self.last_detection_dict:
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time,
                }
                return True

            # Check position change
            last_pos = self.last_detection_dict[class_id]["position"]
            if (
                abs(center_x - last_pos[0]) > self.min_position_change
                or abs(center_y - last_pos[1]) > self.min_position_change
            ):
                # Significant position change
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time,
                }
                return True

            # No sufficient change, don't send
            return False

        # Sufficient time interval, update state and send
        self.last_detection_time = current_time
        self.last_detection_dict[class_id] = {
            "position": (center_x, center_y),
            "time": current_time,
        }
        return True

    def detect(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD)
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            # Process each detection box
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                waste_classifier = WasteClassifier()
                category_id, description = waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"

                color = self.colors.get(class_id, (255, 255, 255))

                # Visualization (if Debug window is enabled)
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                    label = f"{display_text} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.rectangle(
                        frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                # Check if this detection result should be sent
                if self._should_send_detection(
                    class_id, center_x, center_y, confidence
                ):
                    print(f"\nSending detection: {display_text}")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"Bounding box position: ({x1}, {y1}), ({x2}, {y2})")
                    print(f"Center point position: ({center_x}, {center_y})")
                    print("-" * 30)

                    # Directly call original send_to_stm32 method, passing required parameters
                    self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                    self.serial_manager.update_garbage_count(display_text)
                else:
                    print(
                        f"Detected (not sending): {display_text}, position: ({center_x}, {center_y})"
                    )
        return frame


def create_detector(model_path):
    """
    Create YOLODetector instance
    Args:
        model_path: Model file path
    Returns:
        detector: YOLODetector instance
    """
    import os

    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check file extension
    file_extension = os.path.splitext(model_path)[1].lower()
    if file_extension != ".pt":
        raise ValueError(f"Unsupported model format: {file_extension}, only .pt format supported")

    print(f"Using PyTorch model: {model_path}")
    try:
        return YOLODetector(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")


def main():
    use_gpu, device_info = setup_gpu()
    print("\nDevice Information:")
    print(device_info)
    print("-" * 30)

    # Use new detector creation method
    try:
        global model_path
        base_dir = get_script_directory()
        final_path = os.path.join(base_dir, model_path)
        detector = create_detector(final_path)
    except Exception as e:
        print(f"Failed to create detector: {str(e)}")
        return

    cap = find_camera()
    if not cap:
        return

    print("\nSystem Startup:")
    print("- Camera ready")
    print(f"- Debug window: {'Enabled' if DEBUG_WINDOW else 'Disabled'}")
    print(f"- Serial output: {'Enabled' if ENABLE_SERIAL else 'Disabled'}")
    print("- Press 'q' key to exit program")
    print("-" * 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break

            frame = detector.detect(frame)

            if DEBUG_WINDOW:
                window_name = "YOLO_detect"
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nProgram exiting normally")
                    break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, program exiting")
    finally:
        # Clean up resources
        if hasattr(detector, "serial_manager"):
            detector.serial_manager.cleanup()
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
