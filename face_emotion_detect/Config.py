#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
face_model_path = 'models/yolov8n-face.pt'
expression_model_path = 'models/expression_cls.pt'


names = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

colors = [
    '#FF6B6B',  # Angry - 亮珊瑚红
    '#C77DFF',  # Disgust - 亮紫罗兰
    '#FFD166',  # Fear - 亮黄色（原黑色太暗，改为警示黄）
    '#06D6A0',  # Happy - 亮碧绿色
    '#118AB2',  # Neutral - 亮天蓝色
    '#5E72E4',  # Sad - 亮紫蓝色
    '#FF9F1C',  # Surprise - 亮橙色
    '#EF476F',  # 新增1 - 亮品红（比Angry更偏粉）
    '#83C5BE'  # 新增2 - 亮薄荷绿（柔和过渡色）
]

voice_names = {0:'angry',1:'disgusted',2:'fearful',3:'happy',4:'neutral',5:'other',6:'sad',7:'surprised',8:'unk'}




text_names = {0:'anger',1:'disgust',2:'fear',3:'joy',4:'neutral',5:'sadness',6:'shame',7:'surprise'}