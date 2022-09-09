# TensorFlow 2.x Object Detection Configs

Repository of ```.configs``` for the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

The repository aims on having the initial configuration, which lies at [here](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2). In addition, another ```.config``` file with the suffix ```_no_augmentation.config``` exist where all the augmentation is removed. This is done due to the fact that [Roboflow](https://roboflow.com/) includes augmentation is the ```.tfrecord``` file generated from the original dataset. However, all the augmentation options are included in the ```augmentations_all.config``` where one can manually add them in the ```.config``` file. Moreover, ```optimizers_all.config``` includes ADAM and Momentum optimizers where one can manually override in the ```.config``` file.

## 1 Notes

### 1.1 Faster R-CNN
All Faster R-CNN make use of the ```momentum_optimizer``` with ```cosine_decay_learning_rate```. Augmentation includes ```random_horizontal_flip```, ```random_adjust_hue```, ```random_adjust_contrast```, ```random_adjust_saturation```, and ```random_square_crop_by_scale```.

### 1.2 CenterNet (Hourglass)
All CenterNet make use of the ```adam_optimizer``` with ```manual_step_learning_rate```. Augmentation includes ```random_horizontal_flip```, ```random_crop_image```, ```random_adjust_hue```, ```random_adjust_contrast```, ```random_adjust_saturation```, ```random_adjust_brightness```, and ```random_absolute_pad_image```.

### 1.3 EfficientDet
All EfficientDet make use of the ```momentum_optimizer``` with ```cosine_decay_learning_rate```. Augmentation include ```random_horizontal_flip```, and ```random_scale_crop_and_pad_to_square```.

### 1.4 SSD
All SSD (exluded EfficientDet) make use of the ```momentum_optimizer``` with ```cosine_decay_learning_rate```. Augmentation includes random_horizontal_flip, random_crop_image, ssd_random_crop, 

## 2 Optimizers

More options availiable at [TF2 Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/optimizer.proto)

### 2.1 ADAM with Manual Step
```
 optimizer {
    adam_optimizer: {
      epsilon: 1e-7  # Match tf.keras.optimizers.Adam's default.
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 1e-3
          schedule {
           step: 90000
           learning_rate: 1e-4
          }
          schedule {
            step: 120000
            learning_rate: 1e-5
          }
        }
      }
```

### 2.2 ADAM with Cosine Decay
```
  optimizer {
    adam_optimizer: {
      epsilon: 1e-7  # Match tf.keras.optimizers.Adam's default.
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 1e-3
          total_steps: 50000
          warmup_learning_rate: 2.5e-4
          warmup_steps: 5000
        }
      }
    }
    use_moving_average: false
  }
```
### 2.3 Momentum with Cosine Decay
```
optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: .04
          total_steps: 25000
          warmup_learning_rate: .013333
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
## 3 Augemtations

More options availiable at [TF2 Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto)


```
data_augmentation_options {
    random_horizontal_flip {
    }
  }

  data_augmentation_options {
    random_adjust_hue {
    }
  }

  data_augmentation_options {
    random_adjust_contrast {
    }
  }

  data_augmentation_options {
    random_adjust_saturation {
    }
  }

  data_augmentation_options {
    random_adjust_brightness {
    }
  }

  data_augmentation_options {
     random_square_crop_by_scale {
      scale_min: 0.6
      scale_max: 1.3
    }
  }

  data_augmentation_options {
    random_crop_image {
      min_aspect_ratio: 0.5
      max_aspect_ratio: 1.7
      random_coef: 0.25
    }
  }

  data_augmentation_options {
    random_absolute_pad_image {
       max_height_padding: 200
       max_width_padding: 200
       pad_color: [0, 0, 0]
    }
  }

  data_augmentation_options {
    ssd_random_crop {
    }
  }
```

## 4 Metrics

To use mean average precision and recall, you should configure your ```pipeline.config``` file. The metrics_set parameter in the ```eval_config``` block should be set to ```"coco_detection_metrics"```.
```
eval_config: {
 metrics_set: "coco_detection_metrics"
 use_moving_averages: false
 batcH_size: 1;
}
```
Other options incude ```"pascal_voc_detection_metrics"``` and ```"oid_challenge_detection_metrics"```

## 5 Models

The ```.configs``` included concern object detectors that output boxes and masks (no keypoints). One can download the pre-trained weights from the links below and use the ```.configs``` provided in this repository. The downloadable content includes ```.config``` pipelines. However, **all** of the inluded pipelines are same with the ones in here, with the **exception** of the SSD (EfficientDet included). For the Faster R-CNN, Mask R-CNN, and CenterNet, the ```.config``` are a total match.

Model name                                                                                                                                                                  | Speed (ms) | COCO mAP | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :----------: | :-----:
[CenterNet HourGlass104 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz)                    | 70         | 41.9           | Boxes
[CenterNet HourGlass104 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz)               | 197       | 44.5           | Boxes
[CenterNet Resnet50 V1 FPN 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz)     | 27         | 31.2           | Boxes
[CenterNet Resnet101 V1 FPN 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz)     | 34         | 34.2           | Boxes
[CenterNet Resnet50 V2 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz)     | 27         | 29.5           | Boxes
[CenterNet MobileNetV2 FPN 512x512](http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz)     | 6         | 23.4           | Boxes
[EfficientDet D0 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz)                                  | 39         | 33.6           | Boxes
[EfficientDet D1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)                                  | 54         | 38.4           | Boxes
[EfficientDet D2 768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz)                                  | 67         | 41.8           | Boxes
[EfficientDet D3 896x896](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz)                                  | 95         | 45.4           | Boxes
[EfficientDet D4 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz)                              | 133         | 48.5           | Boxes
[EfficientDet D5 1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz)                             | 222         | 49.7           | Boxes
[EfficientDet D6 1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz)                             | 268         | 50.5           | Boxes
[EfficientDet D7 1536x1536](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz)                             | 325         | 51.2           | Boxes
[SSD MobileNet v2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)                                |19         | 20.2           | Boxes
[SSD MobileNet V1 FPN 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz)                        | 48        | 29.1           | Boxes
[SSD MobileNet V2 FPNLite 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)                | 22         | 22.2           | Boxes
[SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)                | 39         | 28.2           | Boxes
[SSD ResNet50 V1 FPN 640x640 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)                          | 46         | 34.3           | Boxes
[SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)                      | 87         | 38.3           | Boxes
[SSD ResNet101 V1 FPN 640x640 (RetinaNet101)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz)                        | 57         | 35.6           | Boxes
[SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)                    | 104        | 39.5           | Boxes
[SSD ResNet152 V1 FPN 640x640 (RetinaNet152)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz)                        | 80         | 35.4           | Boxes
[SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)                    | 111        | 39.6           | Boxes
[Faster R-CNN ResNet50 V1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz)                 | 53         | 29.3           | Boxes
[Faster R-CNN ResNet50 V1 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz)             | 65         | 31.0           | Boxes
[Faster R-CNN ResNet50 V1 800x1333](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz)               | 65         | 31.6           | Boxes
[Faster R-CNN ResNet101 V1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz)               |    55      | 31.8           | Boxes
[Faster R-CNN ResNet101 V1 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz)           | 72         | 37.1           | Boxes
[Faster R-CNN ResNet101 V1 800x1333](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz)             | 77         | 36.6           | Boxes
[Faster R-CNN ResNet152 V1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz)               | 64         | 32.4           | Boxes
[Faster R-CNN ResNet152 V1 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz)           | 85         | 37.6           | Boxes
[Faster R-CNN ResNet152 V1 800x1333](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz)             | 101         | 37.4           | Boxes
[Faster R-CNN Inception ResNet V2 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz)             | 206         | 37.7           | Boxes
[Faster R-CNN Inception ResNet V2 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz)             | 236         | 38.7           | Boxes
[Mask R-CNN Inception ResNet V2 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz) |    301      | 39.0/34.6           | Boxes/Masks
