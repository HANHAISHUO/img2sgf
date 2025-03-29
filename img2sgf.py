import os
import re
import shutil
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import math
from paddleocr import PaddleOCR

x_edge_default = 1311  # 识别后主棋盘的x
y_edge_default = 1311  # 识别后主棋盘的y
chess_diameter = 69  # 棋子直径
main_board_threshold = 1800  # 主棋盘边长阈值
thresh_area = 130 * 2100  # 一行字的面积阈值，用于部署棋盘蒙版
line_space = 300  # 行间距阈值
mask_to_edge = 200  # 边界蒙版宽度
chess_radius = int((chess_diameter - 1) / 2)  # 棋子半径
local_output = './Local Board'  # 局部棋盘输出路径
main_output = './Main Board'  # 主棋盘输出路径
direction_default = False, False, 19, 19 # 用于判断棋盘的方向：四个参数分别为上边界、左边界、纵路数、横路数(True为裁剪边界;False为棋盘边界)
interval = 100 # 查找下一颗棋子前的间隔，0为手动控制

# 将原图转换成边缘检测之后的位图
def img2edge(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)  # 滤波降噪
    img_edge = cv2.Canny(img_blur, 100, 200)  # 边缘检测
    kernel = np.ones((9, 9), np.uint8)  # 膨胀核
    img_edge = cv2.dilate(img_edge, kernel, iterations=1)  # 膨胀
    return img_edge

# 用于清空输出文件夹
def clear_directory(path):
    for filename in os.listdir(path): # 遍历文件夹中的所有文件和文件夹
        file_path = os.path.join(path, filename) # 获取文件或文件夹的完整路径
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) # 如果是文件或链接，则删除
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# 标准化棋盘大小
def compress_image(image, target_size):
    cv2.imwrite("temp.jpg", image)
    image = Image.open("temp.jpg")
    # 新的宽度和高度
    new_width = target_size[0]
    new_height = target_size[1]
    # 调整图像大小
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image.save("temp.jpg")
    image = cv2.imread("temp.jpg")
    return image


# 边缘检测
def edge_detect(board, x_num, y_num):
    up, down, left, right = False, False, False, False
    img = board.copy()
    img_edge = img2edge(img)
    # cv2.imshow('before_edge_detect', img_edge)
    h, w = img_edge.shape[0], img_edge.shape[1]  # 获取图像尺寸
    thresh = int(chess_radius * 2 / 3)  # 边距检测阈值为棋子半径的2/3
    up_avg, down_avg, left_avg, right_avg = 0, 0, 0, 0
    # up 检测
    up_num = 0
    for i in range(0 + chess_diameter, w - chess_diameter):  # 遍历上边所有点(去掉两角)
        num = 0
        while True:
            if img_edge[num][i] == 0:
                num += 1
            else:
                break
        up_avg += num
        if num > thresh:
            up_num += 1
    up_avg = int(up_avg / (w - chess_diameter * 2))
    if up_num < 18:  # 共检测17个边界落子点
        up = True
    # down 检测
    down_num = 0
    for i in range(0 + chess_diameter, w - chess_diameter):  # 遍历下边所有点(去掉两角)
        num = 0
        while True:
            if img_edge[h - 1 - num][i] == 0:
                num += 1
            else:
                break
        down_avg += num
        if num > thresh:
            down_num += 1
    down_avg = int(down_avg / (w - chess_diameter * 2))
    if down_num < 18:
        down = True
    # left 检测
    left_num = 0
    for i in range(0 + chess_diameter, h - chess_diameter):  # 遍历左边所有点(去掉两角)
        num = 0
        while True:
            if img_edge[i][num] == 0:
                num += 1
            else:
                break
        left_avg += num
        if num > thresh:
            left_num += 1
    left_avg = int(left_avg / (h - chess_diameter * 2))
    if left_num < 18:
        left = True
    # right 检测
    right_num = 0
    for i in range(0 + chess_diameter, h - chess_diameter):  # 遍历右边所有点(去掉两角)
        num = 0
        while True:
            if img_edge[i][w - 1 - num] == 0:
                num += 1
            else:
                break
        right_avg += num
        if num > thresh:
            right_num += 1
    right_avg = int(right_avg / (h - chess_diameter * 2))
    if right_num < 18:
        right = True
    up_val, down_val, left_val, right_val = 0, 0, 0, 0
    if up == True and down == True:  # 36对应18个棋子直径，37为18个半，减去平均值是为了让棋盘更加标准
        up_val = int(h / ((y_num - 1) * 2) - up_avg)
        down_val = int(h / ((y_num - 1) * 2) - down_avg)
    if up == True and down == False:
        up_val = int(h / ((y_num - 1) * 2 + 1) - up_avg)
    if up == False and down == True:
        down_val = int(h / ((y_num - 1) * 2 + 1) - down_avg)

    if left == True and right == True:
        left_val = int(w / ((x_num - 1) * 2) - left_avg)
        right_val = int(w / ((x_num - 1) * 2) - right_avg)
    if left == True and right == False:
        left_val = int(w / ((x_num - 1) * 2 + 1) - left_avg)
    if left == False and right == True:
        right_val = int(w / ((x_num - 1) * 2 + 1) - right_avg)
    return up_val, down_val, left_val, right_val


# 识别棋盘
def find_board(image_book):
    book = image_book.copy()  # 保留原图
    book_edge = img2edge(book)  # 转为位图
    # cv2.imshow('book_edge', book_edge) # 显示边缘检测结果
    cv2.imwrite("book_edge.jpg", book_edge)
    contours, hierarchy = cv2.findContours(book_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    # vis = [0 for u in range(0, len(hierarchy[0]))]  # hierarchy[0] 里存放所有的轮廓数据
    max_id = -1
    max_area = -1
    for i in range(0, len(hierarchy[0])):
        if hierarchy[0][i][3] != -1:  # 如果该轮廓有父轮廓，则跳过
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > main_board_threshold or h > main_board_threshold: # 如果边长不合适，跳过
            continue
        if w / h > 1.2 or w / h < 0.8:  # 如果宽高比不合适，则跳过 1.2>19/16>1>16/19>0.8
            continue
        if w * h > max_area:
            max_area = w * h
            max_id = i
    x, y, w, h = cv2.boundingRect(contours[max_id])  # 矩形拟合
    # avg = int((w+h) / 2)
    # print(x, y, w, h)
    # cv2.rectangle(img, (x, y), (x+avg, y+avg), (0, 255, 0), 2)
    crop = book[y:y + h, x:x + w]  # 截取主棋盘
    cv2.imwrite('crop.jpg', crop)  # 显示截取结果
    # 边界上补充棋子的半径
    # cv2.imshow("main_board", crop)
    up_val, down_val, left_val, right_val = edge_detect(crop, 19, 19)  # 获取边界参数
    crop = cv2.copyMakeBorder(crop, up_val, down_val, left_val, right_val, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # 将主棋盘标准化
    crop = compress_image(crop, (x_edge_default, y_edge_default))
    # 存储主棋盘图像
    cv2.imwrite(main_output + "/main_board.jpg", crop)
    return crop


# 查找棋子在棋盘中的位置
def locate(x1, y1, x2, y2, board, direction):
    up, left, x_num, y_num = direction # 获取棋盘参数
    min_dis = 10000
    y_edge, x_edge = board.shape[0:2]
    min_x_cent = -1
    min_y_cent = -1
    x_avg = int((x1 + x2) / 2)  # 棋子中心坐标
    y_avg = int((y1 + y2) / 2)
    for x_point in range(0, x_edge - 1, chess_diameter): # 遍历棋盘上所有点
        for y_point in range(0, y_edge - 1, chess_diameter):
            x_cent = x_point + chess_radius  # 落点坐标
            y_cent = y_point + chess_radius
            dis = math.sqrt(((x_avg - x_cent) ** 2 + (y_avg - y_cent) ** 2))
            if dis < min_dis:  # 更新最近落点
                min_dis = dis
                min_x_cent = x_cent
                min_y_cent = y_cent
    # cv2.circle(board_draw, (min_x_cent, min_y_cent), 2, (255, 0, 0), -1)
    a = int((min_x_cent - chess_radius) / chess_diameter)
    b = int((min_y_cent - chess_radius) / chess_diameter)
    if up == True: # 如果是局部棋盘
        b = 19 - (y_num - b)
    if left == True:
        a = 19 - (x_num - a)
    return a, b


# 棋子计数(以黑棋为参照)
def count_chess(main_board, direction):
    up, left, x_num, y_num = direction
    num_black = 0
    board_draw = main_board.copy()
    vis_count = [[0 for u in range(0, 19)] for v in range(0, 19)] # 记录已经识别过的棋子
    board_gray = cv2.cvtColor(board_draw, cv2.COLOR_BGR2GRAY) # 转为灰度图
    chess = cv2.imread("Pieces Base 2/Black.jpg") # 读取黑棋
    chess_gray = cv2.cvtColor(chess, cv2.COLOR_BGR2GRAY) # 转为灰度图
    match = cv2.matchTemplate(board_gray, chess_gray, cv2.TM_CCOEFF_NORMED) # 模板匹配
    location = np.where(match >= 0.45) # 匹配度大于0.45的棋子
    # print(  location[0].shape[0])
    h, w = chess_gray.shape[0:2] # 获取模板大小
    for pt in zip(*location[::-1]): # 遍历所有匹配点
        x1, y1 = pt[0], pt[1] # 获取棋子左上角坐标
        x2, y2 = pt[0] + w, pt[1] + h # 获取棋子右下角坐标
        a, b = locate(x1, y1, x2, y2, board_draw, direction) # 获取棋子在棋盘中的位置
        if vis_count[a][b] == 1: # 如果该位置已经识别过，则跳过
            continue
        vis_count[a][b] = 1
        # print(a+1,b+1) # 棋子的坐标
        num_black += 1
        cv2.rectangle(board_draw, (x1, y1), (x2, y2), (0, 255, 0), 2) # 画出棋子
    # zoom_show(board_draw)
    # cv2.waitKey(0)
    return num_black * 2

# 画出棋盘模型
def draw_all_point(image):
    for x_point in range(0, x_edge_default, chess_diameter):
        for y_point in range(0, y_edge_default, chess_diameter):
            x_cent = x_point + chess_radius  # 落点坐标
            y_cent = y_point + chess_radius
            cv2.circle(image, (x_cent, y_cent), 2, (0, 0, 255), -1)

# 将图像缩放到合适的尺寸并显示
def zoom_show(board_origin):
    cv2.imwrite("zoom.jpg", board_origin) # 保存图像
    board_zoom = Image.open("zoom.jpg") # 打开图像
    # 计算图像的宽度和高度
    width, height = 437, 437
    # 调整图像大小
    board_zoom = board_zoom.resize((width, height), Image.Resampling.LANCZOS)
    board_zoom.save("zoom.jpg")
    zoom = cv2.imread("zoom.jpg")
    cv2.imshow("board", zoom)


# 初始化黑棋定位棋盘
def initiate_black_board(main_board_image):
    board_kernel1 = np.ones((5, 5), np.uint8)  # 膨胀核
    board_kernel2 = np.ones((3, 3), np.uint8)  # 腐蚀核
    board_kernel3 = np.ones((2, 2), np.uint8)  # 腐蚀核
    black_board_gray = cv2.cvtColor(main_board_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    black_board_blur = cv2.GaussianBlur(black_board_gray, (7, 7), 2)  # 滤波降噪
    black_board_ret, black_board_binary = cv2.threshold(black_board_blur, 150, 255, cv2.THRESH_BINARY)  # 二值化
    black_board_dilation = cv2.dilate(black_board_binary, board_kernel1, iterations=1)  # 膨胀
    black_board_erosion = cv2.erode(black_board_dilation, board_kernel2, iterations=1)  # 腐蚀
    black_board_erosion = cv2.erode(black_board_erosion, board_kernel3, iterations=1)  # 腐蚀
    return black_board_erosion


# 初始化白棋定位棋盘
def initiate_white_board(main_board_image):
    white_board_gray = cv2.cvtColor(main_board_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    white_board_blur = cv2.GaussianBlur(white_board_gray, (17, 17), 3)  # 滤波降噪7,7,2
    return white_board_blur


# 定位黑棋
def find_black_chess(black_board_erosion, chess_image, opt):
    chess_kernel1 = np.ones((5, 5), np.uint8)  # 膨胀核
    chess_kernel2 = np.ones((4, 4), np.uint8)  # 腐蚀核
    chess_gray = cv2.cvtColor(chess_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    chess_blur = cv2.GaussianBlur(chess_gray, (7, 7), 2)  # 滤波降噪
    chess_ret, chess_binary = cv2.threshold(chess_blur, 145, 255, cv2.THRESH_BINARY)  # 二值化145|150
    chess_dilation = cv2.dilate(chess_binary, chess_kernel1, iterations=1)  # 膨胀
    chess_erosion = cv2.erode(chess_dilation, chess_kernel2, iterations=1)  # 腐蚀
    match = cv2.matchTemplate(black_board_erosion, chess_erosion, cv2.TM_CCOEFF_NORMED)  # 模板匹配
    if str(opt) == "max":
        max_match = np.max(match)  # 获取匹配结果
        location = np.where(match == max_match)  # 获取匹配位置
        h, w = chess_gray.shape[0:2]  # 获取模板大小
        pt = next(zip(*location[::-1]))  # 获取匹配坐标
        x1, y1 = pt[0], pt[1]
        x2, y2 = pt[0] + w, pt[1] + h
        return x1, y1, x2, y2
    else:
        chess = [[], [], [], []] # 用于存储匹配到的棋子的坐标
        location = np.where(match > opt) # 获取匹配位置
        w, h = chess_blur.shape[0:2]
        for pt in zip(*location[::-1]): # 遍历匹配到的坐标
            chess[0].append(pt[0])
            chess[1].append(pt[1])
            chess[2].append(pt[0] + w)
            chess[3].append(pt[1] + h)
        return chess


# 定位白棋
def find_white_chess(white_board_erosion, chess_image, opt):
    chess_gray = cv2.cvtColor(chess_image, cv2.COLOR_BGR2GRAY) # 转为灰度图
    chess_blur = cv2.GaussianBlur(chess_gray, (15, 15), 3)  # 滤波降噪5,5,2
    match = cv2.matchTemplate(white_board_erosion, chess_blur, cv2.TM_CCOEFF_NORMED) # 模板匹配
    if str(opt) == "max": # 如果opt为max，则返回匹配结果中置信度最高的位置
        max_match = np.max(match) # 以下同黑棋匹配
        location = np.where(match == max_match)
        h, w = chess_blur.shape[0:2]
        pt = next(zip(*location[::-1]))
        x1, y1 = pt[0], pt[1]
        x2, y2 = pt[0] + w, pt[1] + h
        return x1, y1, x2, y2
    else:
        chess = [[], [], [], []]
        location = np.where(match > opt)
        w, h = chess_blur.shape[0:2]
        for pt in zip(*location[::-1]):
            chess[0].append(pt[0])
            chess[1].append(pt[1])
            chess[2].append(pt[0] + w)
            chess[3].append(pt[1] + h)
        return chess


# 寻找所有没有标号的棋子
def find_all_unordered(black_board, white_board, direction):
    unordered_black_cnt = 0 # 用于记录未标号的黑棋数量
    sgf = "" # 用于存储SGF文件的字符串
    vis_detect = [[0 for u in range(0, 19)] for v in range(0, 19)]  # 保证每个位置只计数一次
    for name in ["BLack", "BLack_delta"]: # 遍历所有未标号黑棋图片
        black_chess = cv2.imread("./Pieces Base 2/" + name + ".jpg")
        if name == "BLack":
            black_pos = find_black_chess(black_board, black_chess, 0.9) # 匹配纯色黑棋，置信阈值为0.9
        else:
            black_pos = find_black_chess(black_board, black_chess, 0.8) # 匹配Δ黑棋，置信阈值为0.8
        if len(black_pos[0]) > 0: # 如果匹配到黑棋
            sgf += "AB"
        for i in range(0, len(black_pos[0])): # 遍历所有匹配到的黑棋
            x1, y1, x2, y2 = black_pos[0][i], black_pos[1][i], black_pos[2][i], black_pos[3][i] # 获取棋子的坐标
            a, b = locate(x1, y1, x2, y2, black_board, direction) # 定位棋子位置
            if vis_detect[a][b] == 1: # 如果该位置已经标记过，则跳过
                continue
            vis_detect[a][b] = 1 # 标记该位置
            unordered_black_cnt += 1 # 未标号黑棋数量加一
            sgf += "[" + chr(a + 1 + 96) + chr(b + 1 + 96) + "]" # 添加到SGF文件中
    for name in ["White", "White_delta"]: # 遍历所有未标号白棋图片
        white_chess = cv2.imread("./Pieces Base 2/" + name + ".jpg")
        if name == "White":
            white_pos = find_white_chess(white_board, white_chess, 0.55) # 匹配纯色白棋，置信阈值为0.55
        else:
            white_pos = find_white_chess(white_board, white_chess, 0.7) # 匹配Δ白棋，置信阈值为0.7
        if len(white_pos[0]) > 0:
            sgf += "AW"
        for i in range(0, len(white_pos[0])): # 遍历所有匹配到的白棋
            x1, y1, x2, y2 = white_pos[0][i], white_pos[1][i], white_pos[2][i], white_pos[3][i] # 获取棋子的坐标
            a, b = locate(x1, y1, x2, y2, white_board, direction)  # 定位棋子位置
            if vis_detect[a][b] == 1: # 如果该位置已经标记过，则跳过
                continue
            vis_detect[a][b] = 1 # 标记该位置
            sgf += "[" + chr(a + 1 + 96) + chr(b + 1 + 96) + "]" # 添加到SGF文件中
    return sgf, unordered_black_cnt # 返回SGF文件字符串和未标号黑棋数量


# 识别图像中的文字
def OCR(image_book):
    pages = image_book.copy()
    h, w = pages.shape[0], pages.shape[1] # 获取图像的宽高
    cv2.rectangle(pages, (0, 0), (w, mask_to_edge), (255, 255, 255), -1) # 以下五行为在图像上下左右中添加白色蒙版
    cv2.rectangle(pages, (0, 0), (mask_to_edge, h), (255, 255, 255), -1)
    cv2.rectangle(pages, (0, h - mask_to_edge), (w, h), (255, 255, 255), -1)
    cv2.rectangle(pages, (w - mask_to_edge, 0), (w, h), (255, 255, 255), -1)
    cv2.rectangle(pages, (int(w / 2) - mask_to_edge, 0), (int(w / 2) + mask_to_edge, h), (255, 255, 255),
                  -1)  # 防止书缝影响拟合
    mask = pages.copy()
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转灰度
    mask_blur = cv2.GaussianBlur(mask_gray, (3, 3), 0)  # 滤波降噪
    mask_edge = cv2.Canny(mask_blur, 100, 200)  # 边缘检测
    kernel = np.ones((9, 9), np.uint8)  # 膨胀核
    mask_edge = cv2.dilate(mask_edge, kernel, iterations=2)  # 膨胀，因为这里不一样所以不放在同一个函数里
    contours, hierarchy = cv2.findContours(mask_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imwrite("del.jpg", mask_edge)
    cv2.waitKey(0)
    for i in range(0, len(hierarchy[0])):
        if hierarchy[0][i][3] != -1:  # 如果该轮廓有父轮廓，则跳过
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        if w * h > thresh_area or h > line_space:  # 如果识别为非文字区域
            cv2.rectangle(pages, (x, y), (x + w, y + h + int(chess_diameter * 1.7)), (255, 255, 255), -1)
    # cv2.imshow("image_book",image_book)
    cv2.imwrite("after_mask.jpg", pages)
    cv2.waitKey(0)
    left_page = pages[0:pages.shape[0], 0:int(pages.shape[1] / 2)] # 将图像分为左右两部分
    right_page = pages[0:pages.shape[0], int(pages.shape[1] / 2):pages.shape[1]]
    cv2.imwrite("left_page.jpg", left_page)
    cv2.imwrite("right_page.jpg", right_page)
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # 初始化PaddleOCR
    result_left = ocr.ocr("left_page.jpg", cls=False)  # 识别整页书
    result_right = ocr.ocr("right_page.jpg", cls=False)  # 识别整页书
    text = ""
    for line in result_left[0]: # 遍历识别结果
        if line[1][1] >= 0.9: # 如果置信度大于0.9
            text += line[1][0] # 添加到文本中
    for line in result_right[0]:
        if line[1][1] >= 0.9:
            text += line[1][0]
    return text


# 将文字与棋子链接起来
def find_comment(text):
    # 使用正则表达式匹配以“黑1”、“白2”开始，并以句号`。`结束的句子
    pattern = r"((白\d{0,2}[02468]|黑\d{0,2}[13579]).*?(。|$))"
    matches = re.findall(pattern, text)
    matches = [list(match) for match in matches]
    # print(matches)
    comments = {} # 创建一个空字典来存储棋子和对应的注释
    for match in matches:
        match[1] = match[1].replace('黑', 'B') # 将“黑”替换为“B”
        match[1] = match[1].replace('白', 'W') # 将“白”替换为“W”
        comments[match[1]] = match[0]
    return comments


# 纵横路数检测
def road(w, h):
    if w > int(chess_diameter * 17.5):  # 如果宽度大于17.5个棋子直径，则认为完整的19路
        x_num = 19
    else:
        x_num = int((w - chess_radius) / chess_diameter) + 1 # 否则，计算宽度可以容纳的棋子数
    if h > int(chess_diameter * 17.5):
        y_num = 19
    else:
        y_num = int((h - chess_radius) / chess_diameter) + 1
    return x_num, y_num


# 棋盘方向:检测线处理
def judge_direction(line, num):
    for i in range(0, 3):
        cnt = 0 # 统计相邻像素颜色不同的个数
        for j in range(0, len(line[i]) - 1):
            if line[i][j] != line[i][j + 1]: # 如果相邻像素颜色不同
                cnt += 1
        if cnt == num * 2: # 识别为局部棋盘
            return True
    return False


# 棋盘方向检测
def direction_detect(board, x_num, y_num):
    up, down, left, right = False, False, False, False
    h, w = board.shape[0], board.shape[1]
    # 上边界检测
    up_line = [[], [], []]
    for i in range(0, 3):
        up_line[i] = [board[int(chess_radius * (i + 1) / 4)][j] for j in range(0, w)] # 取上边界三条检测线
    up = judge_direction(up_line, x_num) # 检测线处理
    # 下边界检测
    down_line = [[], [], []]
    for i in range(0, 3):
        down_line[i] = [board[h - int(chess_radius * (i + 1) / 4)][j] for j in range(0, w)] # 取下边界三条检测线
    down = judge_direction(down_line, x_num) # 检测线处理
    # 左边界检测
    left_line = [[], [], []]
    for i in range(0, 3):
        left_line[i] = [board[j][int(chess_radius * (i + 1) / 4)] for j in range(0, h)] # 取左边界三条检测线
    left = judge_direction(left_line, y_num) # 检测线处理
    # 右边界检测
    right_line = [[], [], []]
    for i in range(0, 3):
        right_line[i] = [board[j][w - int(chess_radius * (i + 1) / 4)] for j in range(0, h)] # 取右边界三条检测线
    right = judge_direction(right_line, y_num) # 检测线处理
    return up, down, left, right

# 寻找局部棋盘
def local_board_find(image_book):
    if os.path.exists(local_output):
        clear_directory(local_output) # 清空文件夹
    else:
        os.makedirs(local_output) # 创建文件夹
    pages = image_book.copy()
    h, w = pages.shape[0], pages.shape[1] # 获取图像的高和宽
    cv2.rectangle(pages, (0, 0), (w, mask_to_edge), (255, 255, 255), -1) # 去除页眉页脚书缝
    cv2.rectangle(pages, (0, h - mask_to_edge), (w, h), (255, 255, 255), -1)
    cv2.rectangle(pages, (int(w / 2) - mask_to_edge, 0), (int(w / 2) + mask_to_edge, h), (255, 255, 255),
                  -1)  # 防止书缝影响拟合
    mask = pages.copy()
    mask_edge = img2edge(mask)  # 膨胀
    contours, hierarchy = cv2.findContours(mask_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 轮廓检测
    max_area = -1
    local_num = 0
    for i in range(0, len(hierarchy[0])):
        if hierarchy[0][i][3] != -1:  # 如果该轮廓有父轮廓，则跳过
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > main_board_threshold or h > main_board_threshold: # 如果边长不合适，跳过
            continue
        if w / h > 1.2 or w / h < 0.8:  # 如果宽高比不合适，则跳过
            continue
        if w * h > max_area:
            max_area = w * h
    for i in range(0, len(hierarchy[0])):
        if hierarchy[0][i][3] != -1:  # 如果该轮廓有父轮廓，则跳过
            continue
        x, y, w, h = cv2.boundingRect(contours[i]) # 获取轮廓的边界框
        if w * h < max_area and w >= chess_diameter * 5 and h >= chess_diameter * 5:  # 如果识别为局部棋盘
            cv2.rectangle(pages, (x, y), (x + w, y + h), (255, 0, 0), 10) # 画框
            local_board = mask[y:y + h, x:x + w] # 裁剪出局部棋盘
            x_num, y_num = road(w, h) # 计算棋盘纵横路数
            up_val, down_val, left_val, right_val = edge_detect(local_board, x_num, y_num)  # 获取边界参数
            local_board = cv2.copyMakeBorder(local_board, up_val, down_val, left_val, right_val, cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255]) # 填充边界
            local_board_edge = img2edge(local_board)
            # 初始化局部棋盘
            sgf_local = "(;FF[4]SZ[19]" # SGF字符串初始化
            vis_local = [[0 for u in range(0, 19)] for v in range(0, 19)]  # 保证每个位置只计数一次
            up, down, left, right = direction_detect(local_board_edge, x_num, y_num) # 获取棋盘方向
            direction = (up, left, x_num, y_num) # 棋盘方向
            local_white_board = initiate_white_board(local_board) # 初始化白棋棋盘
            local_black_board = initiate_black_board(local_board) # 初始化黑棋棋盘
            sgf_unordered_local, cnt = find_all_unordered(local_black_board, local_white_board, direction) # 检测无序号棋子
            total_local = count_chess(local_board, direction) - cnt * 2 # 计算总棋子数
            sgf_local += sgf_unordered_local # 更新sgf字符串
            color = color_judge(local_white_board) # 判断先手
            if color == 1:  # 因为用后手(黑色)判断步数，所以要多判断一手
                total_local += 1
            for i in range(0, total_local):
                board_draw_single = local_board.copy()
                chess_name = ""
                if i % 2 == color:
                    chess_name = "B" + str(i + 1) + ".jpg"
                    chess_image = cv2.imread("./Pieces Base 2/" + chess_name) # 读取黑棋
                    x1, y1, x2, y2 = find_black_chess(local_black_board, chess_image, "max") # 棋子识别
                else:
                    chess_name = "W" + str(i + 1) + ".jpg"
                    chess_image = cv2.imread("./Pieces Base 2/" + chess_name) # 读取白棋
                    x1, y1, x2, y2 = find_white_chess(local_white_board, chess_image, "max") # 棋子识别
                cv2.rectangle(board_draw_single, (x1, y1), (x2, y2), (0, 255, 0), 3) # 画框
                a, b = locate(x1, y1, x2, y2, local_board, direction) # 计算棋子坐标
                if i == total_local - 1 and vis_local[a][b] == 1: # 如果是最后一个棋子且该位置已经有棋子，则跳过
                    break
                vis_local[a][b] = 1
                print("当前落子：" + str(i + 1))
                print(a + 1, b + 1)  # 棋子的坐标
                if i % 2 == color:
                    sgf_local += ";B[" + chr(a + 1 + 96) + chr(b + 1 + 96) + "]" # 更新sgf字符串
                else:
                    sgf_local += ";W[" + chr(a + 1 + 96) + chr(b + 1 + 96) + "]" # 更新sgf字符串

            sgf_local += ")"
            with open(local_output + "/Local Board " + str(local_num + 1) + ".sgf", "w") as l:
                l.write(sgf_local)
            l.close()

            print("Local Board " + str(local_num + 1) + " detected:")
            print("图像大小: " + str((w, h)))
            print("纵横路数: " + str((x_num, y_num)))
            print("截取检测: 顺序为上下左右|True为截取的边界|False为棋盘边界" + str(
                direction_detect(local_board_edge, x_num, y_num)))
            cv2.imwrite(local_output + "/Local Board " + str(local_num + 1) + ".jpg", local_board)
            local_num += 1

# 先手颜色判断
def color_judge(board):
    chess_black = cv2.imread("./Pieces Base 2/B1.jpg") # 读取黑棋
    chess_white = cv2.imread("./Pieces Base 2/W1.jpg") # 读取白棋
    chess_black_gray = cv2.cvtColor(chess_black, cv2.COLOR_BGR2GRAY) # 转换为灰度图
    chess_black_blur = cv2.GaussianBlur(chess_black_gray, (15, 15), 3) # 高斯模糊
    match_black = cv2.matchTemplate(board, chess_black_blur, cv2.TM_CCOEFF_NORMED) # 模板匹配
    max_match_black = np.max(match_black) # 最大匹配值
    chess_white_gray = cv2.cvtColor(chess_white, cv2.COLOR_BGR2GRAY) # 转换为灰度图
    chess_white_blur = cv2.GaussianBlur(chess_white_gray, (15, 15), 3) # 高斯模糊
    match_white = cv2.matchTemplate(board, chess_white_blur, cv2.TM_CCOEFF_NORMED) # 模板匹配
    max_match_white = np.max(match_white) # 最大匹配值
    if max_match_black > max_match_white: # 如果黑棋匹配值大于白棋匹配值，则黑棋先手
        return 0  # 黑棋先手
    else:
        return 1  # 白棋先手


# 主程序
if os.path.exists(main_output):
    clear_directory(main_output) # 清空输出文件夹
else:
    os.mkdir(main_output) # 创建输出文件夹
num_white = 0  # 计数
num_black = 0
vis_detect = [[0 for u in range(0, 19)] for v in range(0, 19)]  # 保证每个位置只计数一次
book_file = filedialog.askopenfilename()  # 获取路径
image_book = cv2.imread(book_file)  # 读入整页书
image_board = find_board(image_book)  # 寻找主棋盘
board_draw = cv2.imread(main_output+"/main_board.jpg")
# 识别文字
words = OCR(image_book) # 保存为一整个字符串
comments = find_comment(words) # 对整个字符串进行处理
print(words)
with open("words.txt", "w") as file:
    file.write(words)
file.close()
# 定位棋子
sgf = "(;FF[4]SZ[19]" # sgf字符串
main_board_image = cv2.imread(main_output+"/main_board.jpg") # 读取主棋盘
white_board = initiate_white_board(main_board_image) # 初始化白棋棋盘
black_board = initiate_black_board(main_board_image) # 初始化黑棋棋盘
sgf_unordered, cnt = find_all_unordered(black_board, white_board, direction_default) # 查找无序号棋子
sgf += sgf_unordered # 添加无序号棋子到sgf字符串
total = count_chess(main_board_image, direction_default) - cnt * 2  # 棋子总数
color = color_judge(white_board) # 判断先手颜色
for i in range(0, total):
    board_draw_single = board_draw.copy()
    chess_name = ""
    if i % 2 == color:
        chess_name = "B" + str(i + 1) + ".jpg"
        chess_image = cv2.imread("./Pieces Base 2/" + chess_name) # 读取黑棋
        x1, y1, x2, y2 = find_black_chess(black_board, chess_image, "max") # 查找黑棋
    else:
        chess_name = "W" + str(i + 1) + ".jpg"
        chess_image = cv2.imread("./Pieces Base 2/" + chess_name) # 读取白棋
        x1, y1, x2, y2 = find_white_chess(white_board, chess_image, "max") # 查找白棋
    cv2.rectangle(board_draw_single, (x1, y1), (x2, y2), (0, 255, 0), 3) # 画框
    a, b = locate(x1, y1, x2, y2, image_board, direction_default) # 定位棋子
    if i == total - 1 and vis_detect[a][b] == 1: # 如果是最后一个棋子且该位置已经有棋子
        break
    vis_detect[a][b] = 1
    print("当前落子：" + str(i + 1))
    print(a + 1, b + 1)  # 棋子的坐标
    if i % 2 == color:
        sgf += ";B[" + chr(a + 1 + 96) + chr(b + 1 + 96) + "]" # 添加黑棋到sgf字符串
        if "B" + str(i + 1) in comments:
            sgf += "C[" + comments["B" + str(i + 1)] + "]" # 添加注释
    else:
        sgf += ";W[" + chr(a + 1 + 96) + chr(b + 1 + 96) + "]" # 添加白棋到sgf字符串
        if "W" + str(i + 1) in comments:
            sgf += "C[" + comments["W" + str(i + 1)] + "]" # 添加注释

    zoom_show(board_draw_single) # 显示棋盘(缩放之后)
    cv2.waitKey(interval) # 等待

sgf += ")"
with open(main_output + "/main_board.sgf", "w") as f:
    f.write(sgf)
f.close()
# 以下为局部棋盘识别
local_board_find(image_book)