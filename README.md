# Exploratory-data-analysis

- example dataset : [self-driving-car dataset](https://universe.roboflow.com/roboflow-gw7yv/self-driving-car/3)

## 0. Coco dataset analysis library
- start example

      # annotation file path
      anno_path = 'Self Driving Car.v3-fixed-small.coco/export/_annotations.coco.json'

      # rgb image directory path
      img_dir_path = 'Self Driving Car.v3-fixed-small.coco/export/'
      
      # instance generation
      coco = CocoAnalysis(anno_path, img_dir_path)



## 1. Image visualization
- Show 5 random images with annotation

      coco.show_random_img_all()
  ![](figs/random_5_images.png)

- Show 5 random images with only specific class annotation

      coco.show_random_img_class(2) # 2 <- class index
  ![](figs/random_5_images_with_car_anno.png)

- Show an image of specific image id

      coco.show_img_id(0) # 0 <- image index
  ![](figs/image_id_0.png)

- Show big bounding boxes

      coco.show_big_annos() 
  ![](figs/big_anno_5_images.png)

- Show small bounding boxes

      coco.show_small_annos()
  ![](figs/small_anno_5_images.png)

## 2. Data chart
- Bar chart of the number of bounding box by class
      
      coco.show_bar_chart_box_by_class()
  ![](figs/bar_chart_box_by_class.png)

- Bar chart of the number of bounding box by image

      coco.show_bar_chart_box_by_image() 
  ![](figs/bar_chart_box_by_image.png)

- Bar chart of the number of bounding box descended by image 

      coco.show_bar_chart_box_by_image_descending()
  ![](figs/bar_chart_box_by_image_descending.png)

- Scatter chart of bounding box height & width

      coco.show_scatter_chart_height_width_all()
  ![](figs/scatter_chart_height_width_all.png)

## 3. Dataset edit



