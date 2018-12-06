 
# Regular extraction. 32 len, 32 step
python patch_extract_2ims.py -ds="para"

# Full extraction. 32 len 32 step
python patch_extract_2ims.py -ds="para" -wpx="any" -at=True -of="patches_full/"


# Example

source="para";
target="acre";


python patch_extract_2ims.py -ds=$source -wpx="any" -at=True -tras=16 -val=True

#python patch_extract_2ims.py -ds=$target -tras=16
python patch_extract_2ims.py -ds=$target -val=True -atst=True

# Train on source
python adda.py -sds=$source -tds=$target -ting=0 -sval=1 -s="results_val/source_weights_acre.h5" -advval=1
python adda.py -sds=$source -tds=$target -ting=0 -sval=1 -advval=1
python adda.py -sds=$source -ting=0

# Evaluate on target
python adda.py -t=True -s="results_val/source_weights_para.h5" -sds $target

# Adversrial train
python adda.py -f -s="results_val/source_weights_para.h5" -sds $source -tds=$target -advval=1 

# Maybe train/test on target
python adda.py -sds=$target -ting=1 -ws=0


# Example2

source="acre";
target="para";

python patch_extract_2ims.py -ds=$source -wpx="any" -at=True -tras=16

python patch_extract_2ims.py -ds=$target -val=True -tras=16

python adda.py -sds=$source -ting=0

# Evaluate on target
python adda.py -t=True -s="results/source_weights_acre.h5" -sds $target

# Adversrial train
python adda.py -f -s="results/source_weights_acre.h5" -sds $source -tds=$target -advval=1
