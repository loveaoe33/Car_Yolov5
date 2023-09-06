import os 
import xml.etree.ElementTree as ET


all_boxes={}
def parse_xml(xml_path):
    tree=ET.parse(xml_path)
    root=tree.getroot()

    boxes=[]

    for obj in root.findall('object'):
        name=obj.find('name').text
        bbox=obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((name, (xmin, ymin, xmax, ymax)))
    return boxes

def convert_all_xmls(xml_dir):
    global all_boxes  # 告訴Python我們要使用全域的all_boxes變數
 
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            xml_path=os.path.join(xml_dir, filename)
            boxes=parse_xml(xml_path)
            all_boxes[filename[:-4]]=boxes
            print(all_boxes)
            box_save_file = os.path.join(xml_dir, filename[:-4] + '.box')
            save_boxes_to_tesseract_format(boxes,box_save_file)
 
    return all_boxes

def save_boxes_to_tesseract_format(boxes, output_file):
    with open(output_file,'w') as f:
       for box in boxes:
            name, (xmin, ymin, xmax, ymax) = box
            width = xmax - xmin
            height = ymax - ymin
            left = xmin
            top = ymin
            line = f"{left} {top} {width} {height} {name} \n"
            f.write(line)
xml_directory="C:\\Users\\loveaoe33\\anaconda3\\envs\\yolov5\\tessdata\\text_train"
box_save_file="C:\\Users\\loveaoe33\\anaconda3\\envs\\yolov5\\tessdata\\output_box_file\\output_box_file.txt"
all_boxes=convert_all_xmls(xml_directory)

print(all_boxes)



# def conver_xml_to_box(xml_file,outfile):
#     # 指定圖像和 XML 文件的目錄路徑
#     Imgae_dir=""
#     xml_dir=""
#     # 建立輸出的 Box 文件的目錄
#     output_dir=""
#     os.makedirs(output_dir,exists=True)

#     for filename in os.listdir(Imgae_dir):
#         if filename.endswith(".tif") or filename.endswith(".jpg") or filename.endswith(".png"):
#             image_file=os.path.join(Imgae_dir,filename)
#             xml_dir=os.path.join(xml_fir,os.path.splitext(filename)[0]+".xml")
            
#             if os.path.isfile(xml_file):
#                 output_dir=os.path.join(output_dir,os.path.splitext(filename)[0]+".box")
#                 conver_xml_to_box(xml_file,output_dir)
#                 print(f"Converted {xml_file} to {output_file}")
#             else:
#                 print(f"No corresponding XML file found for {image_file}")

