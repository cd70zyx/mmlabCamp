# 作者 : 钢之乐学 - 西电
# 功能 : 作业1 - 使用MobileNetV2做图像分类推理 - 基于mmcls.apis

# ======================================
# 下载配置文件
import os
os.system('mim download mmcls --config mobilenet-v2_8xb32_in1k --dest ./models/mobilenet-v2_8xb32_in1k')
# exit()
# ======================================

# 使用API
from mmcls.apis import init_model, inference_model, show_result_pyplot

# 初始化模型
model = init_model(
    './models/mobilenet-v2_8xb32_in1k/mobilenet-v2_8xb32_in1k.py',
    './models/mobilenet-v2_8xb32_in1k/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
    device='cuda:0'
    )
# print(model)

# 推理
path_img = r'data\flower_dataset_split\val\daisy\99306615_739eb94b9e_m.jpg'
result = inference_model(model, path_img)
print(result)

# 显示推理结果
show_result_pyplot(
    model,
    path_img,
    result
    )
