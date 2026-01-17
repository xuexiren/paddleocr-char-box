import os
import cv2
import numpy as np
from paddleocr import PaddleOCR


# 初始化 OCR
ocr = PaddleOCR(
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    return_word_box=False,
)


# ==========================================
# 核心功能函数：单字符切分逻辑
# ==========================================
def get_char_boxes_from_crop(crop_img, text_content):
    h, w = crop_img.shape[:2]
    char_count = len(text_content)
    if char_count == 0:
        return []

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vertical_projection = np.sum(binary, axis=0)

    segments = []
    start_idx = None
    threshold = 0

    for i, val in enumerate(vertical_projection):
        if val > threshold and start_idx is None:
            start_idx = i
        elif val <= threshold and start_idx is not None:
            segments.append((start_idx, i))
            start_idx = None

    if start_idx is not None:
        segments.append((start_idx, w))

    final_char_boxes = []

    # 策略判断
    if len(segments) == char_count:
        for seg in segments:
            final_char_boxes.append([seg[0], 0, seg[1] - seg[0], h])
    else:
        avg_width = w / char_count
        for i in range(char_count):
            x_start = int(i * avg_width)
            x_end = int((i + 1) * avg_width)
            if i == char_count - 1:
                x_end = w
            final_char_boxes.append([x_start, 0, x_end - x_start, h])

    return final_char_boxes



# ==========================================
# 主流程 (修改为单张图片处理)
# ==========================================

if __name__ == "__main__":
    # 1. 图片路径
    img_path = input("请输入图片的路径：")
    if not os.path.exists(img_path):
        print("错误：文件不存在，请检查路径。")
        exit()

    # 2. 读取图片
    original_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if original_img is None:
        print("错误：无法读取图片，请检查文件格式。")
        exit()

    vis_img = original_img.copy() # 用于画图的副本

    print(f"正在处理图片：{os.path.basename(img_path)} ...")

    try:
        # 3. PaddleOCR 检测
        results = ocr.predict(input=img_path)
        if results is None:
            print("没有检测到字符")

        for result_group in results:
            if result_group is None:
                continue
            res = result_group.json["res"]
            rec_boxes = res["rec_boxes"]
            rec_texts = res["rec_texts"]
            for i, box in enumerate(rec_boxes):
                x_min, y_min, x_max, y_max = box
                text = rec_texts[i]
                # 1. 裁剪
                y_min = max(0, y_min)
                x_min = max(0, x_min)
                crop_img = original_img[y_min:y_max, x_min:x_max]
                if crop_img.size == 0:
                    continue   


                # 2. 核心切分：获取单字符框 (相对坐标)
                char_boxes_relative = get_char_boxes_from_crop(crop_img, text)

                # 5. 坐标还原并画框
                for idx, char_box in enumerate(char_boxes_relative):
                    if idx >= len(text): 
                        break

                    cx, cy, cw, ch = char_box

                    # 还原绝对坐标 = 裁剪框左上角 + 字符相对坐标
                    abs_x = int(x_min + cx)
                    abs_y = int(y_min + cy)
                    abs_w = int(cw)
                    abs_h = int(ch)

                    # 画红色矩形框 (BGR: 0, 0, 255)
                    cv2.rectangle(vis_img, (abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h), (0, 0, 255), 1)

        # 6. 生成输出路径并保存
        dir_name = os.path.dirname(img_path)
        file_name = os.path.basename(img_path)
        name, ext = os.path.splitext(file_name)
        
        # 输出文件名：原文件名_result.jpg
        output_path = os.path.join(dir_name, f"{name}_result{ext}")
        
        cv2.imencode(ext, vis_img)[1].tofile(output_path)
        print(f"处理成功！结果已保存至：{output_path}")

    except Exception as e:
        print(f"处理过程中发生错误：{e}")