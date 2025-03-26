import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import os
from tkinter import Tk, filedialog
import json
import datetime
import uuid

# Load model SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = r"D:\SAM\sam_vit_b_01ec64.pth"  # Đường dẫn đến trọng số model
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)

# Biến để lưu danh sách các hình ảnh và chỉ số hiện tại
image_paths = []
current_image_idx = 0

# Đường dẫn để lưu mask và thông tin
save_dir = r"D:\SAM\saved_masks"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Thêm các biến toàn cục để lưu mẫu màu từ ảnh trước
previous_hat_dieu_colors = []
previous_hat_dieu_sizes = []
previous_nen_colors = []
previous_nen_sizes = []

# Thêm các biến toàn cục để lưu tỷ lệ co của ảnh
scale_x = 1.0
scale_y = 1.0

# Hàm để tìm kiếm và tải mẫu màu từ bất kỳ ảnh nào có mask trong thư mục
def find_and_load_masks_from_any_image():
    global hat_dieu_colors, hat_dieu_sizes, nen_colors, nen_sizes
    global previous_hat_dieu_colors, previous_hat_dieu_sizes, previous_nen_colors, previous_nen_sizes
    global image, masks_for_current_image
    
    found_mask = False
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".json"):
                metadata_path = os.path.join(root, file)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        image_path_from_metadata = metadata["image_path"]
                        if os.path.exists(image_path_from_metadata):
                            print(f"Đã tìm thấy mask từ ảnh {os.path.basename(image_path_from_metadata)} trong thư mục {root}")
                            temp_image = cv2.imread(image_path_from_metadata)
                            if temp_image is None:
                                print(f"Không thể đọc ảnh: {image_path_from_metadata}")
                                continue
                            original_image = image
                            image = temp_image
                            predictor.set_image(image)  # Cập nhật predictor với ảnh tạm
                            loaded_masks = load_saved_masks(image_path_from_metadata)
                            temp_masks = masks_for_current_image
                            masks_for_current_image = loaded_masks
                            success = load_samples_from_saved_masks()
                            masks_for_current_image = temp_masks
                            image = original_image
                            if success:
                                print(f"Đã tải mẫu màu từ ảnh {os.path.basename(image_path_from_metadata)}")
                                found_mask = True
                                break
                except Exception as e:
                    print(f"Lỗi khi tải mask từ metadata {metadata_path}: {e}")
        if found_mask:
            break
    
    if not found_mask:
        print("Không tìm thấy mask trong bất kỳ thư mục con nào.")
        return False
    
    # Cập nhật predictor với ảnh hiện tại sau khi tải mẫu màu
    current_path = image_paths[current_image_idx]
    current_image = cv2.imread(current_path)
    if current_image is not None:
        image = current_image.copy()
        predictor.set_image(image)
        print(f"Đã cập nhật predictor với ảnh hiện tại: {current_path}, shape: {image.shape}")
    else:
        print(f"Lỗi: Không thể đọc lại ảnh hiện tại {current_path}")
    
    return True

# Hàm để load mẫu màu từ ảnh trước
def load_samples_from_previous_image():
    global hat_dieu_colors, hat_dieu_sizes, nen_colors, nen_sizes
    global previous_hat_dieu_colors, previous_hat_dieu_sizes, previous_nen_colors, previous_nen_sizes

    # Kiểm tra xem có mẫu màu nào đã được lưu từ ảnh trước
    if previous_hat_dieu_colors and previous_nen_colors:
        # Sử dụng các mẫu màu hạt điều và nền từ ảnh trước
        hat_dieu_colors = previous_hat_dieu_colors.copy()
        hat_dieu_sizes = previous_hat_dieu_sizes.copy()
        nen_colors = previous_nen_colors.copy()
        nen_sizes = previous_nen_sizes.copy()

        print("Đã nạp mẫu hạt điều/nền từ ảnh trước")
        return True
    else:
        print("Không có mẫu màu từ ảnh trước")
        return False

# Hàm để chọn thư mục chứa ảnh
def select_folder():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ Tkinter chính
    folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
    root.destroy()
    
    if not folder_path:  # Nếu người dùng hủy việc chọn thư mục
        return False
        
    global image_paths, current_image_idx
    # Lấy tất cả file ảnh trong thư mục
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and 
                  any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not image_paths:
        print("Không tìm thấy ảnh trong thư mục!")
        return False
    
    # Sắp xếp lại danh sách dựa trên số trong tên tệp
    def extract_number(filename):
        # Trích xuất số từ tên tệp (ví dụ: từ "0.jpg" lấy 0)
        import re
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(0)) if match else 0
    
    image_paths.sort(key=extract_number)
    
    current_image_idx = 0
    print(f"Đã tìm thấy {len(image_paths)} ảnh trong thư mục")
    
    # Tự động load ảnh đầu tiên
    load_current_image()
    
    return True
# Hàm để load mẫu màu từ mask đã lưu
def load_samples_from_saved_masks():
    global hat_dieu_colors, hat_dieu_sizes, nen_colors, nen_sizes
    
    if not masks_for_current_image:
        print("Không có mask đã lưu cho ảnh này!")
        return False

    # Kích thước của ảnh hiện tại (resize ảnh nếu cần)
    image_height, image_width = image.shape[:2]
    
    # Lọc mask theo kích thước để phân biệt hạt điều và nền
    # Sắp xếp mask theo kích thước tăng dần
    sorted_masks = sorted(masks_for_current_image, key=lambda x: x.get("mask_size", 0))
    
    # Lấy 1/3 số mask có kích thước nhỏ nhất làm mẫu cho hạt điều
    mask_sizes = [m.get("mask_size", 0) for m in masks_for_current_image]
    median_size = np.median(mask_sizes) if mask_sizes else 0
    hat_dieu_masks = [m for m in masks_for_current_image if m.get("mask_size", 0) < median_size]
    if not hat_dieu_masks:
        hat_dieu_masks = sorted_masks[:max(1, len(sorted_masks) // 3)]
    
    # Reset các biến lưu trữ
    hat_dieu_colors = []
    hat_dieu_sizes = []
    nen_colors = []
    nen_sizes = []
    
    # Lấy mẫu cho hạt điều từ các mask nhỏ
    for mask_data in hat_dieu_masks:
        mask = mask_data.get("mask")
        mask_size = mask_data.get("mask_size", 0)

        # Resize mask về kích thước của ảnh hiện tại
        if mask is not None:
            resized_mask = cv2.resize(mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            y, x = np.where(resized_mask == 1)
            if len(y) <= 5:
                continue
            if len(y) > 0:
                sample_indices = np.random.choice(len(y), min(10, len(y)), replace=False)
                for idx in sample_indices:
                    if 0 <= y[idx] < image.shape[0] and 0 <= x[idx] < image.shape[1]:
                        pixel_color = image[y[idx], x[idx]]
                        hat_dieu_colors.append(pixel_color)
                        hat_dieu_sizes.append(mask_size)

    # Lấy mẫu cho nền từ khu vực ngoài các mask
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    for mask_data in masks_for_current_image:
        mask = mask_data.get("mask")
        if mask is not None:
            resized_mask = cv2.resize(mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            combined_mask = combined_mask | resized_mask
    
    non_mask_y, non_mask_x = np.where(~combined_mask)
    if len(non_mask_y) > 0:
        sample_indices = np.random.choice(len(non_mask_y), min(20, len(non_mask_y)), replace=False)
        for idx in sample_indices:
            if 0 <= non_mask_y[idx] < image.shape[0] and 0 <= non_mask_x[idx] < image.shape[1]:
                pixel_color = image[non_mask_y[idx], non_mask_x[idx]]
                nen_colors.append(pixel_color)
                nen_sizes.append(0)

    print(f"Đã tự động tải {len(hat_dieu_colors)} mẫu màu hạt điều và {len(nen_colors)} mẫu màu nền")
    global previous_hat_dieu_colors, previous_hat_dieu_sizes, previous_nen_colors, previous_nen_sizes
    previous_hat_dieu_colors = hat_dieu_colors.copy()
    previous_hat_dieu_sizes = hat_dieu_sizes.copy()
    previous_nen_colors = nen_colors.copy()
    previous_nen_sizes = nen_sizes.copy()

    return len(hat_dieu_colors) > 0 and len(nen_colors) > 0

# Hàm để load và hiển thị ảnh hiện tại
def load_current_image():
    global image, hat_dieu_colors, hat_dieu_sizes, nen_colors, nen_sizes, masks_for_current_image, mode, scale_x, scale_y, predictor
    
    # Reset các biến lưu trữ thông tin về hạt điều và nền
    hat_dieu_colors = []
    hat_dieu_sizes = []
    nen_colors = []
    nen_sizes = []
    masks_for_current_image = []  # Reset danh sách mask cho ảnh mới
    
    # Kiểm tra xem có ảnh nào được chọn chưa
    if not image_paths:
        print("Chưa có ảnh nào được chọn. Vui lòng chọn thư mục ảnh trước!")
        return None
    
    # Lấy đường dẫn ảnh hiện tại
    current_path = image_paths[current_image_idx]
    print(f"Đang mở ảnh: {current_path}")
    
    # Đọc và xử lý ảnh
    original_image = cv2.imread(current_path)
    if original_image is None:
        print(f"Lỗi: Không thể đọc ảnh {current_path}")
        return None
    original_height, original_width = original_image.shape[:2]

    image = original_image.copy()
    scale_x = scale_y = 1.0
    predictor.set_image(image)  # Đảm bảo predictor được cập nhật với ảnh mới
    print(f"Đã đặt predictor cho ảnh mới, shape: {image.shape}")
    
    # Tải mask đã lưu cho ảnh hiện tại
    masks_for_current_image = load_saved_masks(current_path)
    if masks_for_current_image:
        print(f"Đã tìm thấy {len(masks_for_current_image)} mask đã lưu cho ảnh này")
        if load_samples_from_saved_masks():
            print("Đã tải mẫu màu từ mask, sẵn sàng segment")
            mode = "segment"
    else:
        print("Không có mask đã lưu cho ảnh này!")
        if find_and_load_masks_from_any_image():
            print("✅ Đã tìm thấy và nạp mẫu màu từ thư mục con, sẵn sàng segment!")
            mode = "segment"
        else:
            print("⚠ Không tìm thấy mẫu màu trong thư mục con, yêu cầu chọn thủ công!")
            mode = "select_hat_dieu"
    
    return image

# Hàm để kiểm tra xem điểm click đã nằm trong mask đã lưu chưa
def is_mask_already_saved(current_path, click_x, click_y):
    # Tải các mask đã lưu cho ảnh hiện tại
    saved_masks = load_saved_masks(current_path)
    
    # Kiểm tra xem điểm click có nằm trong mask đã lưu không
    for saved_mask in saved_masks:
        mask = saved_mask.get("mask")
        if mask is not None:
            # Lấy các chỉ số pixel của mask (các vị trí có giá trị 1 trong mask)
            y, x = np.where(mask == 1)  # Tọa độ của các pixel có giá trị 1 trong mask
            
            # Kiểm tra xem điểm click có nằm trong các chỉ số này không
            if (click_y in y) and (click_x in x):
                print("Điểm click đã nằm trong mask đã lưu.")
                return True
    
    return False

def generate_unique_mask_filename(base_path, base_filename, timestamp, unique_id):
    # Tạo tên tệp duy nhất bằng cách kết hợp base_filename, timestamp và unique_id
    unique_filename = f"{base_filename}_{timestamp}_{unique_id}.npy"
    return os.path.join(base_path, unique_filename)

# Hàm kiểm tra tên tệp và tạo tên duy nhất nếu tệp đã tồn tại
def get_unique_filename(file_path):
    count = 1
    base_file_path = file_path
    while os.path.exists(base_file_path):
        # Nếu tệp đã tồn tại, tạo một tên tệp mới bằng cách thêm số đếm
        base_file_path = f"{os.path.splitext(file_path)[0]}_{count}{os.path.splitext(file_path)[1]}"
        count += 1
    return base_file_path
# Ngăn xếp lưu trữ các mask đã thay đổi (theo kiểu ngăn xếp LIFO)
mask_stack = []
redo_stack = []  # Ngăn xếp để lưu trữ các mask đã bị xóa hoặc undo
# Hàm để lưu mask và thông tin liên quan
def save_mask(image_path, mask, mask_size, contour_points):
    # Tạo tên file từ đường dẫn ảnh
    image_name = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name)[0]
    
    # Tạo thư mục cho ảnh này nếu chưa tồn tại
    image_save_dir = os.path.join(save_dir, image_name_without_ext)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    
    # Tạo tên tệp duy nhất và lưu mask
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Tạo ID ngẫu nhiên từ uuid, lấy 8 ký tự đầu tiên
    mask_filename = f"mask_{timestamp}"
    mask_path = generate_unique_mask_filename(image_save_dir, mask_filename, timestamp, unique_id)
    
    # Kiểm tra xem tên tệp có bị trùng không, nếu có, tạo tên mới
    mask_path = get_unique_filename(mask_path)
    
    # Lưu mask dưới dạng numpy array (chuyển tất cả giá trị >0 về 1)
    mask = (mask > 0).astype(np.uint8)
    np.save(mask_path, mask)
    
    # Lưu thông tin metadata
    metadata_path = os.path.splitext(mask_path)[0] + ".json"  # Tạo metadata_path từ mask_path
    
    metadata = {
        "image_path": image_path,
        "mask_path": mask_path,  # Lưu đường dẫn tệp mask
        "mask_size": int(mask_size),
        "timestamp": timestamp,
        "contour_points": [c.tolist() for c in contour_points],
        "metadata_path": metadata_path  # Lưu metadata_path vào metadata
    }
    
    # Lưu metadata vào file JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Đã lưu mask và thông tin tại: {mask_path}")
    
    # Thêm mask vào mask_stack với dữ liệu đầy đủ (bao gồm mask)
    metadata["mask"] = mask
    mask_stack.append(metadata)
    
    return metadata


# Hàm để tải các mask đã lưu cho một ảnh
def load_saved_masks(image_path):
    image_name = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name)[0]
    
    image_save_dir = os.path.join(save_dir, image_name_without_ext)
    if not os.path.exists(image_save_dir):
        return []
    
    mask_metadatas = []
    
    # Tìm tất cả file json trong thư mục
    for file in os.listdir(image_save_dir):
        if file.endswith(".json"):
            metadata_path = os.path.join(image_save_dir, file)
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Tải mask từ file numpy
                    mask_path = metadata["mask_path"]
                    if os.path.exists(mask_path):
                        metadata["mask"] = np.load(mask_path)
                        # print(f"🟢 Đọc mask từ file {mask_path} - Kiểu dữ liệu: {metadata['mask'].dtype}, Kích thước: {metadata['mask'].shape}")
                        mask_metadatas.append(metadata)
            except Exception as e:
                print(f"Lỗi khi tải mask: {e}")
    
    return mask_metadatas

# Hàm để hiển thị các mask đã lưu
def display_saved_masks():
    # Vẽ tất cả các mask đã lưu trực tiếp lên ảnh gốc
    if not masks_for_current_image:
        print("Không có mask đã lưu cho ảnh này")
        return
    
    # Vẽ tất cả các mask đã lưu
    for mask_data in masks_for_current_image:
        mask = mask_data.get("mask")
        if mask is not None:
            # Lấy contour từ mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Vẽ contour với màu xanh lá cây lên ảnh gốc
            color = (0, 255, 0)  # Màu xanh lá cây
            cv2.drawContours(image, contours, -1, color, 2)  # Vẽ các contour với màu xanh lá cây

# Initialize other variables
hat_dieu_colors = []  # Lưu danh sách màu của hạt điều
hat_dieu_sizes = []   # Lưu danh sách kích thước mask hạt điều
nen_colors = []  # Lưu danh sách màu của nền
nen_sizes = []   # Lưu danh sách kích thước mask nền
mode = "select_hat_dieu"  # Chế độ: "select_hat_dieu" hoặc "select_nen" hoặc "segment"
masks_for_current_image = []  # Lưu danh sách các mask đã lưu cho ảnh hiện tại
# Thêm 2 biến toàn cục để lưu màu và kích thước hạt điều cho tất cả ảnh
global_hat_dieu_colors = []
global_hat_dieu_sizes = []
# Khởi tạo biến image
image = None

def get_avg(lst):
    """Tính trung bình danh sách"""
    return np.mean(lst, axis=0) if len(lst) > 0 else None

def get_color_distance(color1, color2):
    """Tính khoảng cách Euclidean giữa 2 màu trong không gian RGB"""
    return np.linalg.norm(np.array(color1) - np.array(color2))
def filter_and_clean_mask(mask, min_contour_points=100):
    """Lọc và giữ lại mask liên thông lớn nhất"""
    # Tìm các contour trong mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nếu không có contour nào tìm thấy, trả về mask trống
    if len(contours) == 0:
        return np.zeros_like(mask)

    # Tìm contour lớn nhất (theo diện tích)
    largest_contour = max(contours, key=cv2.contourArea)

    # Tạo một mask mới chỉ chứa contour lớn nhất
    cleaned_mask = np.zeros_like(mask)
    cv2.drawContours(cleaned_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return cleaned_mask

def on_click(event, x, y, flags, param):
    global mode, hat_dieu_colors, hat_dieu_sizes, nen_colors, nen_sizes
    global previous_hat_dieu_colors, previous_hat_dieu_sizes, previous_nen_colors, previous_nen_sizes
    global global_hat_dieu_colors, global_hat_dieu_sizes
    global scale_x, scale_y, image

    # Kiểm tra xem có ảnh nào được chọn chưa
    if not image_paths or image is None:
        print("Chưa có ảnh nào được chọn hoặc ảnh hiện tại không hợp lệ. Vui lòng chọn thư mục ảnh trước!")
        return

    current_path = image_paths[current_image_idx]
    print(f"Click trên ảnh: {current_path}, shape: {image.shape}")

    if event == cv2.EVENT_LBUTTONDOWN:
        # Chuyển đổi tọa độ click về tọa độ gốc của ảnh
        original_x = int(x * (1 / scale_x))
        original_y = int(y * (1 / scale_y))
        print(f"Tọa độ click (gốc): ({original_x}, {original_y})")

        # Kiểm tra xem người dùng muốn xóa mask
        if mode == "delete_mask":
            delete_mask(original_x, original_y)
            return
        
        # Kiểm tra xem điểm click có nằm trong mask đã lưu không
        if is_mask_already_saved(current_path, original_x, original_y):
            print("Mask này đã được lưu rồi, không cần lưu nữa.")
            return
        
        # Kiểm tra xem pixel tại tọa độ click có hợp lệ không
        if 0 <= original_y < image.shape[0] and 0 <= original_x < image.shape[1]:
            pixel = image[original_y, original_x]  # Lấy màu pixel tại tọa độ gốc
        else:
            print(f"Tọa độ click ({original_x}, {original_y}) ngoài phạm vi ảnh!")
            return

        if mode in ["select_hat_dieu", "select_nen"]:
            input_point = np.array([[x, y]])
            input_label = np.array([1])

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            
            mask = masks[0]
            mask_size = np.sum(mask)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
                mask_size = np.sum(mask)

            if mode == "select_hat_dieu":
                hat_dieu_colors.append(pixel)
                hat_dieu_sizes.append(mask_size)
                print(f"Đã chọn hạt điều - Màu: {pixel}, Kích thước: {mask_size}")

            elif mode == "select_nen":
                nen_colors.append(pixel)
                nen_sizes.append(mask_size)
                print(f"Đã chọn nền - Màu: {pixel}, Kích thước: {mask_size}")

        elif mode == "segment":
            avg_hat_dieu_color = get_avg(hat_dieu_colors)
            avg_hat_dieu_size = get_avg(hat_dieu_sizes)
            avg_nen_color = get_avg(nen_colors)
            avg_nen_size = get_avg(nen_sizes)

            if avg_hat_dieu_color is None or avg_nen_color is None:
                print("Vui lòng chọn màu và kích thước của hạt điều & nền trước khi segment!")
                return

            color_distance_to_hat_dieu = get_color_distance(avg_hat_dieu_color, pixel)
            color_distance_to_nen = get_color_distance(avg_nen_color, pixel)

            input_point = np.array([[x, y]])
            input_label = np.array([1])

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )

            mask = masks[0]
            mask_size = np.sum(mask)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
                mask_size = np.sum(mask)

            size_threshold_min = avg_hat_dieu_size * 0.1
            size_threshold_max = avg_hat_dieu_size * 3

            print(f"Khoảng cách màu đến hạt điều: {color_distance_to_hat_dieu:.2f}, đến nền: {color_distance_to_nen:.2f}, Kích thước mask: {mask_size}")

            if color_distance_to_nen < color_distance_to_hat_dieu:
                print(f"Bỏ qua vì màu {pixel} thuộc về nền (khoảng cách đến nền nhỏ hơn)!")
                return

            if mask_size < size_threshold_min or mask_size > size_threshold_max:
                print(f"Bỏ qua vì kích thước {mask_size} không phù hợp! Màu: {pixel}")
                return

            cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
            predictor.set_image(image)

            metadata = save_mask(current_path, mask, mask_size, [largest_contour])
            if metadata:
                metadata["mask"] = mask
                masks_for_current_image.append(metadata)
                display_saved_masks()
            
            global_hat_dieu_colors.extend(hat_dieu_colors)
            global_hat_dieu_sizes.extend(hat_dieu_sizes)

def delete_mask(x, y):
    global masks_for_current_image, mask_stack, redo_stack
    deleted_count = 0  # Biến đếm số lượng mask bị xóa

    for i, mask_data in enumerate(masks_for_current_image):
        mask = mask_data.get("mask")
        if mask is not None:
            # Lấy các chỉ số pixel của mask (các vị trí có giá trị 1 trong mask)
            y_coords, x_coords = np.where(mask == 1)

            # Kiểm tra xem điểm click có nằm trong mask không
            if (y in y_coords) and (x in x_coords):
                print(f"Đã xóa mask tại vị trí ({x}, {y})")

                # Lưu đường dẫn của tệp mask và metadata trước khi xóa
                mask_path = mask_data.get("mask_path")
                metadata_path = mask_data.get("metadata_path")

                # Kiểm tra xem metadata_path có phải là None không
                if metadata_path is None:
                    print("Lỗi: metadata_path không tồn tại.")
                    return

                # Kiểm tra sự tồn tại của file metadata trước khi xóa
                if not os.path.exists(metadata_path):
                    print("Không thể tìm thấy file metadata, không thể xóa.")
                    return

                # Lưu mask vào redo_stack trước khi xóa
                redo_stack.append(mask_data)
                print(f"Đã thêm mask vào redo_stack: {mask_path}")

                # Xóa mask khỏi masks_for_current_image
                del masks_for_current_image[i]

                # Xóa mask khỏi mask_stack nếu tồn tại
                mask_stack[:] = [m for m in mask_stack if m["mask_path"] != mask_path]

                # Xóa tệp mask và metadata khỏi thư mục
                if os.path.exists(mask_path):
                    os.remove(mask_path)  # Xóa tệp mask (npy)
                    print(f"Đã xóa tệp mask: {mask_path}")

                if os.path.exists(metadata_path):
                    os.remove(metadata_path)  # Xóa tệp metadata (json)
                    print(f"Đã xóa tệp metadata: {metadata_path}")

                # Tăng biến đếm số lượng mask đã xóa
                deleted_count += 1

                # Cập nhật lại hiển thị
                load_current_image()  # Tải lại ảnh để cập nhật
                display_saved_masks()  # Hiển thị lại ảnh với các mask đã lưu còn lại

                # Sau khi xóa một mask, thoát khỏi vòng lặp để chỉ xóa 1 mask
                break

    # In số lượng mask đã xóa
    print(f"Số lượng mask bị xóa: {deleted_count}")

def change_mode(new_mode):
    global mode
    mode = new_mode
    print(f"Chế độ hiện tại: {mode}")

    if mode == "delete_mask":
        print("Chế độ xóa mask đã được chọn, nhấn vào mask để xóa.")
    else:
        print(f"Chế độ {mode} đã được chọn")
current_image_idx = 0  # Đảm bảo gán giá trị mặc định khi bắt đầu
# Chuyển đến ảnh tiếp theo
def next_image():
    global current_image_idx
    if current_image_idx is None:
        current_image_idx = 0

    if not image_paths or len(image_paths) <= 1:
        print("Không có ảnh tiếp theo")
        return
    
    current_image_idx = (current_image_idx + 1) % len(image_paths)
    load_current_image()  # Load ảnh mới và reset tất cả trạng thái
    load_samples_from_previous_image()  # Nạp mẫu từ ảnh trước (nếu có)
    display_saved_masks()  # Hiển thị mask của ảnh mới
    print(f"Đã chuyển đến ảnh: {image_paths[current_image_idx]}, shape: {image.shape}") 

# Chuyển đến ảnh trước đó
def prev_image():
    global current_image_idx
    if not image_paths or len(image_paths) <= 1:
        print("Không có ảnh trước đó")
        return
    
    current_image_idx = (current_image_idx - 1) % len(image_paths)
    load_current_image()

def undo_last_mask():
    global mask_stack, redo_stack, masks_for_current_image
    if mask_stack:
        # Lấy mask cuối cùng từ ngăn xếp
        last_mask_metadata = mask_stack.pop()
        mask_path = last_mask_metadata["mask_path"]
        metadata_path = last_mask_metadata["metadata_path"]
        
        # Lưu mask vào redo_stack trước khi xóa
        redo_stack.append(last_mask_metadata)
        print(f"Đã thêm mask vào redo_stack: {mask_path}")
        
        # Xóa mask khỏi ảnh hiện tại
        if os.path.exists(mask_path):
            os.remove(mask_path)
            print(f"Đã xóa mask tại: {mask_path}")
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"Đã xóa metadata tại: {metadata_path}")
        
        # Cập nhật lại danh sách masks_for_current_image
        masks_for_current_image = [m for m in masks_for_current_image if m["mask_path"] != mask_path]
        
        # Cập nhật lại ảnh sau khi xóa mask
        load_current_image()
        display_saved_masks()  # Hiển thị lại ảnh với các mask đã lưu còn lại
    else:
        print("Không có mask nào để xóa.")

def redo_last_mask():
    global redo_stack, mask_stack, masks_for_current_image
    if redo_stack:
        last_mask_metadata = redo_stack.pop()
        mask_path = last_mask_metadata["mask_path"]
        metadata_path = last_mask_metadata["metadata_path"]
        mask = last_mask_metadata["mask"]
        
        # Lưu mask vào tệp .npy
        np.save(mask_path, mask)
        print(f"Đã khôi phục mask tại: {mask_path}")
        
        # Tạo một bản sao của metadata mà không có trường "mask"
        metadata_to_save = {key: value for key, value in last_mask_metadata.items() if key != "mask"}
        
        # Lưu metadata vào tệp JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        print(f"Đã khôi phục metadata tại: {metadata_path}")
        
        # Thêm lại metadata vào mask_stack và masks_for_current_image
        mask_stack.append(last_mask_metadata)
        masks_for_current_image.append(last_mask_metadata)
        
        # Cập nhật lại ảnh
        load_current_image()
        display_saved_masks()
    else:
        print("Không có mask nào để khôi phục.")
