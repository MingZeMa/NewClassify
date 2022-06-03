"""
@Function: onnx模型的生成与运行

@Time:2022-3-14

@Author:马铭泽
"""

import torch.onnx
import onnx
from _3ClassifyNet import *
from _2Dataset_Load import *
import onnxruntime as ort
from _1TrainMain import UsingNet

#生成onnx
def createOnnx():
    # 选取运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 输入模型地址
    FullModelPath = "Output\model"
    # 实例化网络
    if UsingNet == "Net1":
        # TODO:记着换了模型可能要改输出个数
        Net = ClassifyNet(In_Channels=1, Out_Channels=9, Features=6,device=device,file_path=None).to(device)
        Net.load_state_dict(torch.load(os.path.join(FullModelPath, '1.0_0080.pt'), map_location=device))

        Net.eval()

    if UsingNet == "Net2":
        # TODO:记着换了模型可能要改输出个数
        Net = HitNet(In_Channels=1, Out_Channels=9, Features=7).to(device)
        # 导入模型
        Net.load_state_dict(torch.load(os.path.join(FullModelPath, '1.0_0080.pt'), map_location=device))

        Net.eval()

    # input
    Input_Tensor = torch.randn([1, 1, 36, 48]).to(device)
    input_names = ["input_0"]
    output_names = ["output_0"]
    out_path = "6.15_1.0.onnx"

    # start
    torch.onnx.export(
        Net,
        Input_Tensor,# TODO:这个东西突然有点迷
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={'input_0': [2, 3]}
    )


# 运行并测试onnx
def runOnnx(Input, ModelName):
    """
    :param Input: 用于测试onnx的图像
    :return: 输出结果
    """
    #run
    Model = onnx.load(ModelName)
    onnx.checker.check_model(Model)
    Ort_session = ort.InferenceSession(ModelName)
    Outputs = Ort_session.run(None,{'input_0':Input})

    return Outputs

if __name__ == "__main__":
    Create = True
    if Create:
        # 创建onnx
        createOnnx()
    else:
        # 测试onnx
        OnnxName = "7.6_1.3.onnx"
        TestPath = r"Dataset\testImg"

        # 初始化模型
        Model = onnx.load(OnnxName)
        onnx.checker.check_model(Model)
        Ort_session = ort.InferenceSession(OnnxName)


        # 读取图片
        Imgs = os.listdir(TestPath)
        for img in Imgs:
            Img = cv.imread(os.path.join(TestPath, img))
            Output = Ort_session.run(None, {'input_0': Img})
            print("Output", Output)
            print("Class", list(Output).index(max(Output)))
            ShowImg = cv.resize(Img, (640, 480))
            cv.putText(ShowImg, 'Predict:' + str(max(Output)), (120, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 1)
            cv.imshow("Img", Img)
            cv.witKey(1000)


