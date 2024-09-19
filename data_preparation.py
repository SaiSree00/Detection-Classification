from PIL import Image
import pandas as pd


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

def cc_yolo(bb, image_w, image_h):
    x1,y1,x2,y2 = bb
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

# im=Image.open(img_path)
# w= int(im.size[0])
# h= int(im.size[1])


# print(xmin, xmax, ymin, ymax) #define your x,y coordinates
# b = (xmin, xmax, ymin, ymax)
# bb = convert((w,h), b)


import glob

image_paths = glob.glob('data/images/*')


kk = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 94: 2, 24: 3, 18: 4, 13: 5, 26: 6, 27: 6, 15: 7, 31: 8, 48: 8, 62: 8, 65: 8, 68: 8, 69: 8, 74: 8, 75: 8, 81: 8, 84: 8, 86: 8, 32: 9, 29: 10, 33: 10, 37: 10, 49: 10, 30: 11, 44: 11, 66: 11, 87: 11, 89: 11, 91: 11, 61: 12, 79: 13, 34: 14, 41: 14, 52: 14, 35: 15, 36: 16, 78: 16, 38: 17, 39: 17, 71: 17, 72: 17, 88: 17, 42: 18, 45: 19, 70: 19, 47: 20, 51: 20, 54: 20, 58: 20, 60: 20, 80: 20, 83: 20, 96: 20, 22: 21, 63: 21, 85: 21, 56: 22, 57: 22, 64: 22, 77: 22, 50: 23, 59: 23, 67: 23, 76: 23}


ss ={  0: ('Maize', [1, 2, 3, 4, 5, 6]),
    1: ('Sugar beet', [7, 8, 9, 10, 11, 12]),
    2: ('Soy',  [94]),
    3: ('Sunflower', [24]),
    4: ('Potato',  [18]),
    5: ('Pea',  [13]),
    6: ('Bean', [26, 27]),
    7: ('Pumpkin',  [15]),
    8: ('Grasses',  [31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86]),
    9: ('Amaranth', [32]),
    10: ('Goosefoot',  [29, 33, 37, 49]),
    11: ('Knotweed',  [30, 44, 66, 87, 89, 91]),
    12: ('Corn spurry',  [61]),
    13: ('Chickweed',  [79]),
    14: ('Solanales',  [34, 41, 52]),
    15: ('Potato weed', [35]),
    16: ('Chamomile',  [36, 78]),
    17: ('Thistle',  [38, 39, 71, 72, 88]),
    18: ('Mercuries',  [42]),
    19: ('Geranium',  [45, 70]),
    20: ('Crucifer',  [47, 51, 54, 58, 60, 80, 83, 96]),
    21: ('Poppy',  [22, 63, 85]),
    22: ('Plantago', [56, 57, 64, 77]),
    23: ('Labiate',  [50, 59, 67, 76]) }

from tqdm import tqdm 

for i in tqdm(image_paths):
    try:
        label_path = i.replace('/images/','/yolo_labels/')
        label_path = label_path.replace('.'+label_path.split('.')[-1] , '.txt')
        
        lb_path = i.replace('/images/','/bboxes/CropAndWeed/')
        
        lb_path = lb_path.replace('.'+lb_path.split('.')[-1] , '.csv')

        df = pd.read_csv(lb_path , header = None , encoding='utf-8')
        df = df.values

        im=Image.open(i)
        w= int(im.size[0])
        h= int(im.size[1])

        file = open(label_path,'w')

        for j in df:
            b = j[:4]
            if int(j[4]) not in kk:
                continue
            cls = kk[j[4]]
            bb = cc_yolo(b, w, h)
            #convert((w,h), b)
            bb.insert(0,cls)
            bb = list(map(str,bb))
            bb = ' '.join(bb)
            file.write(bb+'\n')

        file.close()
    except Exception as e:
        print(e)
        print(lb_path)








