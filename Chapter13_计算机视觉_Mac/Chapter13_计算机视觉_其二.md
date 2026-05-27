#### 13.3 目标检测和边界框
- 目标检测（Object Detection）/目标识别（Object Recognition）：对于图像中的多个目标，希望知道其类型并得到在图像中的位置。


```python
import torch
from d2l import torch as d2l
```


```python
d2l.set_figsize()
img = d2l.plt.imread('./img/catdog.jpg')
d2l.plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x317d41c00>




    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_2_1.svg)
    


- **边界框**：在目标检测中使用边界框（Bounding Box）来描述对象的空间位置。


```python
#@save
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```


```python
# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```


```python
boxes = torch.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```




    tensor([[True, True, True, True],
            [True, True, True, True]])




```python
#@save
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```


```python
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_8_0.svg)
    


#### 13.4 描框
一种目标检测方法：以每个像素为中心，生成多个缩放比和宽高比不同的边界框，即描框（Anchor Box）。
##### 13.4.1 生成多个描框
假设输入图像的高度为h，宽度为w，以图形的每个像素中心生成不同形状的描框：缩放比（Scale）$s\in[-,1]$，宽高比（Aspect Ratio）$r>0$，则描框的宽度、高度分别为$hs\sqrt{r}$和$\frac{hs}{\sqrt{r}}$。在实践中，只考虑包含$s_1$或$r_1$的组合：
$$(s_1,r_1),(s_1,r_2),\ldots,(s_1,r_m),(s_2,r_1),(s_3,r_1),\ldots,(s_n,r_1)$$
以同一像素为中心的描框的数量是$n+m-1$，对于整个输入图像，共生成$wh(n+m-1)$个描框。

该方法使用```multibox_prior()```函数实现，指定输入图像、缩放比列表和宽高比列表，返回所有描框。


```python
#@save
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```


```python
img = d2l.plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

    561 728





    torch.Size([1, 2042040, 4])




```python
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```




    tensor([0.0551, 0.0715, 0.6331, 0.8215])



定义```show_bboxes()```实现在图像绘制多个边界框。


```python
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```


```python
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1',
             's=0.5,  r=1',
             's=0.25, r=1',
             's=0.75, r=2',
             's=0.75, r=0.5'])
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_15_0.svg)
    


##### 13.4.2 交并比（IoU）
为评价描框对目标的覆盖程度，引入杰卡德相似指数（Jaccard Index）/杰卡德相似系数（Jaccard Similarity Coefficient），以度量描框和真是边界框之间的相似性：
$$J(A,B)=\frac{|A\cap B|}{|A\cup B}$$
对于两个边界框的杰卡德指数成为交并比（Intersection over Union）


```python
#@save
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

##### 13.4.3 在训练数据中标注描框
在训练集中，视每个描框为一个训练样本。为训练目标检测模型，需要每个描框的类别（Class），即描框相关的目标类别，和偏移量（Offeset）标签，即描框与真实边界的偏移量。预测时，为每张图像生成多个描框，预测所有描框的边界框，最后只输出符合特定条件的预测边界框。
1. 将真实的边界框分配个描框

给定图像，假设描框是$A_1,A_2,\ldots,A_{n_a}$，$B_1,B_2,\ldots,B_{n_b}$，真实边界框$n_a\geq n_b$，定义矩阵$\mathbf{X}\in\mathbb{R}^{n_a\times n_b}$，$x_{ij}$是描框$A_i$和真实边界框$B_j$的IoU，计算步骤：
    - 在矩阵$\mathbf{X}$找到最大元素$x_{i_1,j_1}$，即描框与真实边界框最接近的位置，将$B_{j_1}$分配给$A_{i_1}$，丢弃第$i_1$行和$j_1$的元素；
    - 重复上述步骤，直到丢弃$\mathbf{X}$中$n_b$列所有元素，此时已经为$n_b$个描框各自分配了一个真实边界框；
    - 遍历剩余$n_a-n_b$个描框，$\forall A_i$，在$\mathbf{X}$第$i$行找到与$A_i$的IoU最大的$B_j$，当且仅当IoU大于阈值时，将$B_j$分配给$A_i$；
    - 以```assign_anchor_to_bbox()```实现该算法。
    - 真实框（Ground Truth, GT）；描框（Anchors）


```python
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    # 初始化匹配表 anchors_bbox_map：0, -1, 1表示GT0，未分配，GT1
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, 
                                  device=device)    
    # 根据阈值，决定是否分配真实边界框 —— 阈值法
    # 大于阈值iou_threshold，算作正样本
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    # 强制匹配：每个GT至少强行分配一个最佳anchor，保证每个真实目标一定被检测到
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # 找到X中最大的元素x_{ij}
        box_idx = (max_idx % num_gt_boxes).long() # 返回B_j，GT
        anc_idx = (max_idx / num_gt_boxes).long() # 返回A_i，anchor
        anchors_bbox_map[anc_idx] = box_idx       # 强制匹配，必须被认领
        jaccard[:, box_idx] = col_discard         # 删除i行
        jaccard[anc_idx, :] = row_discard         # 删除j列
    return anchors_bbox_map
```

2. 标注类别和偏移量

接下来为每个描框分配标注类别和偏移量。假设一个描框$A$被分配了一个真实边界框$B$，$A$将被标注为与$B$相同，同时$A$的偏移量会根据$B$和$A$中心坐标的相对位置以及两个框的相对大小进行标注。鉴于数据集中不同框的位置和大小不同，可以对相对位置和大小应用变化，获得更均匀易拟合的偏移量。给定$A$和$B$，中心坐标$(x_a,y_a)$和$(x_b,y_b)$，宽高$(w_a,h_a)$和$(w_b,h_b)$，将$A$的偏移量标注为
$$\left(\frac{\frac{x_b-x_a}{w_a}-\mu_x}{\sigma_x},\frac{\frac{y_b-y_a}{h_a}-\mu_y}{\sigma_y},\frac{\log\frac{w_b}{w_a}-\mu_w}{\sigma_w},\frac{\log\frac{h_b}{h_a}-\mu_h}{\sigma_h}\right)$$
其中常设$\mu_x=\mu_y=\mu_w=\mu_h=0$，$\sigma_x=\sigma_y=0.1$，$\sigma_w=\sigma_h=0.2$，在```offset_boxes()```中实现。


```python
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset
```

如果描框没有被分配真实边界框，则将其标注为背景（Background），即负类描框，其余描框为正类描框，使用```multibox_target()```实现。


```python
#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

3. 示例

对于catdog.jpg，定义真实边界框和描框。


```python
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_25_0.svg)
    


其中，背景、狗、猫的索引值为0、1、2，再为描框和真实边界框分配一个维度。


```python
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

第三个元素```labels[2]```表示输入描框的类别，其中$A_1$标注为狗，$A_2$和$A_4$标注为猫，其余低于阈值，标注为背景。


```python
labels[2]
```




    tensor([[0, 1, 2, 0, 2]])



第二个元素```labels[1]```为掩码(Mask)，形状为(batch_size,4*num_anchors)，与描框的4个偏移量一一对应。负类目标不影响目标函数。通过元素乘法，掩码中的零再计算目标函数之前过滤掉负类偏移量。


```python
labels[1]
```




    tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
             1., 1.]])



第一个元素```labels[0]```表示每个描框标注的4个偏移量，负类描框的偏移量标注为0。


```python
labels[0]
```




    tensor([[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,  1.4000e+00,
              1.0000e+01,  2.5940e+00,  7.1754e+00, -1.2000e+00,  2.6882e-01,
              1.6824e+00, -1.5655e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
             -0.0000e+00, -5.7143e-01, -1.0000e+00,  4.1723e-06,  6.2582e-01]])



##### 13.4.4 使用非极大值抑制预测边界框
预测时，先为图像生成多个描框，预测类别和偏移量。使用```offset_inverse()```，将描框和偏移量预测作为输入，应用偏移变换返回预测的边界框坐标。


```python
def offset_inverse(anchors, offset_preds): #@save
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = np.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = np.concatenate((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

对于多个重叠的预测边界框时，未来简化输出，使用非极大值抑制（Non-Maximum Suppression，NMS）合并属于同一目标的相似的预测边界框。引入置信度（Confidence）$p$，对同一图像生成列表$L$，将$p$按降序排序。
- 从$L$选取置信度最高的预测边界框$B_1$作为基准，将所有与$B_1$的IoU超过阈值$\epsilon$的非基准边界框从$L$移除，即具有非极大值置信度的边界框被抑制了；
- 重复上述过程，直至$L$中的任意一对预测边界框的IoU都小于$\epsilon$；
- 输出$L$中的所有预测边界框。


```python
#@save
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)
```

定义```multibox_detection()```来将非极大值抑制应用于预测边界框。
- 分类概率：
```cls_probs```
- 边界框偏移量：```offset_preds```
- 描框：```anchors```
- 输出——最终检测框：类别，置信度，边界框坐标。
- 网络输出->每个```anchor```选最大类别->利用```offset```修正```anchor```位置->```NMS```去重->```置信度过滤```->输出最终框


```python
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)    # 去掉第一维
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]   # 获取类别和anchor数
    out = []
    for i in range(batch_size):     # 遍历batch
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)    # 取第i张图片预测
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
```

示例


```python
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
```


```python
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_42_0.svg)
    


#### 13.5 多尺度目标检测
##### 13.5.1 多尺度描框
在输入图像中均匀抽样一部分像素，以此为中心生成描框。使用较小的描框检测较小的目标，可以抽样较多区域，使用较大的描框检测较大的目标，可以抽样较少区域。


```python
img = d2l.plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]
h, w
```




    (561, 728)



定义```display_anchors()```，在特征图```fmap```生成描框```anchors```，每个像素作为描框的中心。


```python
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```


```python
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_47_0.svg)
    



```python
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_48_0.svg)
    



```python
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```


    
![svg](Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_files/Chapter13_%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89_%E5%85%B6%E4%BA%8C_49_0.svg)
    


##### 13.5.2 多尺度检测
- 基于卷积神经网络的多尺度目标检测：

假设有$c$张$h\times w$的特征图，生成$h\cdot w$组描框，每组有$a$个中心相同的描框。当不同层的特征图在输入图像中分别拥有不同大小的感受野时，可以用于检测不同大小的目标。可以设计一个神经网路，靠经输出层的特征图单元具有更大的感受野，则可以在输入图像中检测较大的目标，即利用深层神经网络在多个层次上对图像进行分层表示，从而实现多尺度目标检测。
