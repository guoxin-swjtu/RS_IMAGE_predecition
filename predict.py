import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 50).__str__()
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import mapping
import fiona
import gc
from osgeo import gdal
from segformer import SegFormer_Segmentation
from utils.utils import cvtColor, preprocess_input, show_config
import colorsys
import copy
import time
import torch
import torch.nn.functional as F
from PIL import Image,ImageChops

from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 30000000000
from torch import nn
from nets.segformer import SegFormer
from osgeo import osr

model = SegFormer_Segmentation()

input_folder = "/media/xsar/F/planet_basemap_download/mountain-merge/small_basemap/"

tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]


for tif_file in tif_files:
    image_path = os.path.join(input_folder, tif_file)

    # 使用GDAL获取地理坐标信息
    image_ds = gdal.Open(image_path)
    geo_transform = image_ds.GetGeoTransform()  # 获取地理变换参数
    projection = image_ds.GetProjection()  # 获取投影信息

    # 读取影像数据
    image = Image.open(image_path)

    # 定义切分参数
    tile_size = (256, 256)  # 每个小块的大小
    stride = 256  # 移动步长，用于重叠区域

    # 获取原始影像的大小
    image_width, image_height = image.size

    # 创建一个空的结果数组
    prediction_list1 = []
    prediction_list2 = []
    prediction_list3 = []

    # 遍历图像并进行切分和预测
    for y in range(0, image_height, stride):
        print("开始预测第%s行" % int(y/256+1))
        for x in range(0, image_width, stride):
            # 提取当前小块
            tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))
            prediction1 = model.detect_image(tile)
            gray_image1 = prediction1
            gray_image1 = prediction1.convert('L')
            
            pred2 = tile.transpose(Image.FLIP_LEFT_RIGHT)     #水平翻转
            prediction2 = model.detect_image(pred2)
            gray_image2 = prediction2.convert('L')
            gray_image2 = gray_image2.transpose(Image.FLIP_LEFT_RIGHT)
            
            pred3 = tile.transpose(Image.FLIP_TOP_BOTTOM)    #垂直翻转
            prediction3 = model.detect_image(pred3)
            gray_image3 = prediction3.convert('L')
            gray_image3 = gray_image3.transpose(Image.FLIP_TOP_BOTTOM)
            
            prediction_list1.append((x, y, gray_image1))
            prediction_list2.append((x, y, gray_image2))
            prediction_list3.append((x, y, gray_image3))
    
    print("预测完成，开始保存预测结果影像")
    output_path1 = os.path.join("/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/test/1.tif")
    output_path2 = os.path.join("/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/test/2.tif")
    output_path3 = os.path.join("/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/test/3.tif")
    
    # 在循环结束后，将预测结果拼接为一个大图像
    output_image1 = Image.new('L', (image_width, image_height))
    output_image2 = Image.new('L', (image_width, image_height))
    output_image3 = Image.new('L', (image_width, image_height))

    for x, y, prediction_tile in prediction_list1:
        output_image1.paste(prediction_tile, (x, y))
    for x, y, prediction_tile in prediction_list2:
        output_image2.paste(prediction_tile, (x, y))
    for x, y, prediction_tile in prediction_list3:
        output_image3.paste(prediction_tile, (x, y))

    # 将预测结果影像保存为GeoTIFF，保留原始地理坐标信息
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path1, image_width, image_height, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(geo_transform)  # 设置地理变换参数
    output_ds.SetProjection(projection)  # 设置投影信息
    output_ds.GetRasterBand(1).WriteArray(np.array(output_image1))
    output_ds = None

    print("预测结果已保存为GeoTIFF文件:", output_path1)
    
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path2, image_width, image_height, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(geo_transform)  # 设置地理变换参数
    output_ds.SetProjection(projection)  # 设置投影信息
    output_ds.GetRasterBand(1).WriteArray(np.array(output_image2))
    output_ds = None

    print("预测结果已保存为GeoTIFF文件:", output_path2)
    
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path3, image_width, image_height, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(geo_transform)  # 设置地理变换参数
    output_ds.SetProjection(projection)  # 设置投影信息
    output_ds.GetRasterBand(1).WriteArray(np.array(output_image3))
    output_ds = None

    print("预测结果已保存为GeoTIFF文件:", output_path3)
    ####################################################################################################
# 输入影像文件路径
    file_path_band1 = output_path1
    file_path_band2 = output_path2
    file_path_band3 = output_path3

# 打开影像文件
    band1_ds = gdal.Open(file_path_band1, gdal.GA_ReadOnly)
    band2_ds = gdal.Open(file_path_band2, gdal.GA_ReadOnly)
    band3_ds = gdal.Open(file_path_band3, gdal.GA_ReadOnly)

    # 读取波段数据
    band1_array = band1_ds.GetRasterBand(1).ReadAsArray()
    band2_array = band2_ds.GetRasterBand(1).ReadAsArray()
    band3_array = band3_ds.GetRasterBand(1).ReadAsArray()

    # 将波段像素值相加
    sum_array = band1_array + band2_array + band3_array

    # 创建输出影像
    
    output_path = "/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/test/4.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path, image_width, image_height, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(geo_transform)  # 设置地理变换参数
    output_ds.SetProjection(projection)  # 设置投影信息
    # output_ds.GetRasterBand(1).WriteArray(np.array(output_image3))
    output_ds.GetRasterBand(1).WriteArray(np.array(sum_array))

    # 关闭数据集
    band1_ds = None
    band2_ds = None
    band3_ds = None
    output_ds = None

    print("影像相加完成，结果保存在", output_path)
    
    ######################################################################################################
    
    
    
    print("开始将其保存成0-255的灰度图像")
    # 打开tif图像
    input_image_path = output_path

    output_path5 = "/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/test/5.tif"
    input_ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    input_array = input_ds.ReadAsArray()# 读取输入图像数据为NumPy数组
    input_array[input_array <= 50] = 0
    # 创建输出图像
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path5, image_width, image_height, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(geo_transform)# 设置地理信息
    output_ds.SetProjection(projection)
    output_ds.GetRasterBand(1).WriteArray(np.array(input_array))# 将NumPy数组写入输出图像
    input_ds = None
    output_ds = None

    print(f"已保存处理后的图像为 {output_path5}")
    
    
    #################################################################################################################
    
    
    
    print("开始提取影像中的边界信息")

    # 读取灰度图像
    image_path = output_path5
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化图像（将像素值为0的设为0，其他设为255）
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # 查找边界
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取原始影像的坐标信息
    src_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    spatial_ref = osr.SpatialReference(wkt=projection)
    crs = spatial_ref.ExportToProj4()

    # 创建一个GeoDataFrame来保存边界信息
    geometries = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 4:  # 过滤掉小于4个坐标点的轮廓
            polygon_coords = [tuple(point[0]) for point in contour]
            
            # 转换坐标点为地理坐标
            polygon_geo_coords = []
            for x, y in polygon_coords:
                lon = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                lat = geotransform[3] + x * geotransform[4] + y * geotransform[5]
                polygon_geo_coords.append((lon, lat))

            polygon = Polygon(polygon_geo_coords)
            geometries.append(polygon)

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)  # 使用原始影像的坐标系

    # 保存为Shapefile
    output_shapefile = "/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/test/test.shp"
    # output_shapefile = os.path.join("/media/xsar/F/guo-x/rock_seg/segformer-pytorch/pre_dection_result/b0-2/",f"{os.path.splitext(tif_file)[0]}_boundary.shp")
    gdf.to_file(output_shapefile)

    print(f"边界已提取并保存为 {output_shapefile}")
    gc.collect()


print('done,请用qgis完成合并和筛选')
    
    
    
        
