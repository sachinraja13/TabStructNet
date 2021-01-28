# TabStructNet
## Table Structure Recognition using Top-Down and Bottom-Up Cues


This code is developed using the code from 
1. https://github.com/matterport/Mask_RCNN
1. https://github.com/shahrukhqasim/TIES-2.0 

```
Download necessary files. For download locations and links please refer to note.txt files in the following folders:
1. https://github.com/sachinraja13/TabStructNet/tree/master/coco_model/coco
2. https://github.com/sachinraja13/TabStructNet/tree/master/trained_model/tab/annotations
3. https://github.com/sachinraja13/TabStructNet/tree/master/trained_model/tab/logs/tab20200821T0923
```

```
To train the model using MS coco weights, execute:
python samples/tabnet/tabnet.py train --dataset=trained_model/tab --model=coco
```

```
To train the model using most recently saved weights, execute:
python samples/tabnet/tabnet.py train --dataset=trained_model/tab --model=last
```

```
To evaluate the model using most recently saved weights, execute:
python samples/tabnet/tabnet.py evaluate --dataset=trained_model/tab --model=last
```

Saved weights provided in the Google Drive link are trained using SciTSR dataset.
UNLV train and test split is added in the repository for easy fine-tuning on UNLV and testing.

'''
To generate output XML:
Execute the TabStructNet model for evaluation as specified in the repository's README.
Copy the 4 result folders generated in the trained_model/tab directory to the results folder inside the rename_output_files folder.
Execute rename_maskrcnn_result_files.py
Copy the 4 result folders generated inside rename_output_files/rename_results to xml_generating_postprocessor directory.
Copy the validation JPEG images inside xml_generating_postprocessor/gt_without_box folder.
Execute cell_postprocessor_adj.py 
XMLs are generated in processed_xmls folder,
'''

##### Please refer to https://github.com/matterport/Mask_RCNN initially for any issues in running the script.


Please use this to cite our work:
```
@misc{raja_2020,
  title={Table Structure Recognition using Top-Down and Bottom-Up Cues},
  author={Sachin Raja, Ajoy Mondal, C V Jawahar},
  year={2020},
  publisher={Springer Science+Business Media},
  journal={Accepted to ECCV-6007}
}
```

## References:
* Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow; Waleed Abdulla; 2017; https://github.com/matterport/Mask_RCNN

* Schreiber, S., Agne, S., Wolf, I., Dengel, A., Ahmed, S.: DeepDeSRT: Deep learning for detection and structure recognition of tables in document images. In: ICDAR. (2017)

* Qasim, S.R., Mahmood, H., Shafait, F.: Rethinking table parsing using graph neural networks. In: ICDAR. (2019)

* Tensmeyer, C., Morariu, V., Price, B., Cohen, S., Martinezp, T.: Deep splitting and merging for table structure decomposition. In: ICDAR. (2019)

* Shahab, A., Shafait, F., Kieninger, T., Dengel, A.: An open approach towards the  benchmarking of table structure recognition systems. In: DAS. (2010)

* Chi, Z., Huang, H., Xu, H.D., Yu, H., Yin, W., Mao, X.L.: Complicated table structure recognition. arXiv (2019)

* Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., Li, Z.: TableBank: Table benchmark for image-based table detection and recognition. In: ICDAR. (2019)
