cd val_json_generator/
rm instances_val2014.json
rm -rf val2014/
rm -rf xml/
python test_images.py
python make_json_for_cells.py
cd ..
rm -rf trained_model/tab/val2014/
rm -f trained_model/tab/annotations/instances_val2014.json
cp -r val_json_generator/val2014 trained_model/tab/
cp val_json_generator/instances_val2014.json trained_model/tab/annotations
rm rename_output_files/GT_json_file/instances_val2014.json
rm -rf rename_output_files/rename_results/result_*
rm -rf rename_output_files/results/result_*
cp val_json_generator/instances_val2014.json rename_output_files/GT_json_file
rm -rf xml_generating_postprocessor/processed_*
rm -rf xml_generating_postprocessor/result_*
rm -rf xml_generating_postprocessor/gt_without_box/*
cp val_json_generator/val2014/* xml_generating_postprocessor/gt_without_box/
python setup.py install
python samples/tabnet/tabnet.py evaluate --dataset=trained_model/tab --model=last
cp -r trained_model/tab/result_* rename_output_files/results/
cd rename_output_files
python rename_maskrcnn_result_files.py
cd ..
cp -r rename_output_files/rename_results/result_* xml_generating_postprocessor
cd xml_generating_postprocessor
python cell_postprocessor_adj.py
cd ..
