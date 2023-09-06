import os
import xml.etree.ElementTree as ET

def convert_xml_to_tesseract_box(xml_file, image_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find('filename').text
    image_path = os.path.join(image_folder, image_name)

    with open(xml_file.replace('.xml', '.txt'), 'w') as f:
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x_min = bbox.find('xmin').text
            y_min = bbox.find('ymin').text
            x_max = bbox.find('xmax').text
            y_max = bbox.find('ymax').text
            class_name = obj.find('name').text

            # 將座標格式轉換為Tesseract-OCR的Box文件格式
            tesseract_box_line = f"{x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max} {class_name}\n"

            f.write(tesseract_box_line)

# 使用示例
xml_directory = "C:\\Users\\loveaoe33\\anaconda3\\envs\\yolov5\\tessdata\\text_train"
image_folder = "C:\\Users\\loveaoe33\\anaconda3\\envs\\yolov5\\tessdata\\text_train"  # 圖片的目錄與XML目錄相同

for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        xml_path = os.path.join(xml_directory, filename)
        convert_xml_to_tesseract_box(xml_path, image_folder)
