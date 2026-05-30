import torch
from thop import clever_format, profile
from nets.yolo import YoloBody

if __name__ == '__main__':
    input_shape = [640, 640]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80
    phi = 's'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YoloBody(input_shape, num_classes, phi, pretrained=False).to(device)
    for i in m.children():
        print(i)
        print('==========================')

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m.to(device), inputs=(dummy_input,), verbose=False)

    # ----------------------------------------------- #
    # thop ne compte chaque convolution que comme une
    # seule opération ; on multiplie par 2 pour avoir
    # les FLOPs réels (multiply + add = 2 ops par MAC).
    # Certains frameworks ne multiplient pas — on choisit
    # la convention x2 pour rester comparable aux papers.
    # ----------------------------------------------- #

    flops = flops * 2
    flops, params = clever_format(nums=[flops, params], format="%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
