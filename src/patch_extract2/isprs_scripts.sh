KERAS_BACKEND=tensorflow



# Example

source="area3"; target="area23";
source="area23"; target="area3";


python patch_extract_2ims.py -ds=$source -at=True -tras=32 -val=True -c=3;
python patch_extract_2ims.py -ds=$target -val=True -atst=True -c=3;


#python patch_extract_2ims.py -ds=$target -val=True -atst=True -sp="scaler_area3" -c=3;

#python patch_extract_2ims.py -ds=$target -tras=16

# Train on source
python adda.py -sds=$source -tds=$target -c=3 -ibcknd=0 -ting=0 -sval=1 -s="results/source_weights_"$source".h5" -advval=1
python adda.py -sds=$source -tds=$target -c=3 -cln=4 -ting=0 -sval=1 -advval=1 -em='basic'


# Evaluate on target
python adda.py -t=True -c=3 -cln=4 -s="results_val/source_weights_"$source".h5" -sds $target 

# Adversrial train
python adda.py -f -c=3 -cln=4 -s="results/source_weights_"$source".h5" -sds $source -tds=$target -advval=1 

# Maybe train/test on target
python adda.py -sds=$target -ting=1 -ws=0


# Example2

source="acre"; target="para";

python patch_extract_2ims.py -ds=$source -wpx="any" -at=True -tras=16 -val=True; 

python patch_extract_2ims.py -ds=$target -val=True -atst=True;

#python patch_extract_2ims.py -ds=$target -val=True -tras=16


python adda.py -sds=$source -tds=$target -ting=0 -sval=1 -advval=1 -s="results_val/source_weights_acre.h5"

# Evaluate on target
python adda.py -t=True -s="results_val/source_weights_acre.h5" -sds $target

# Adversrial train
python adda.py -f -s="results_val/source_weights_acre.h5" -sds $source -tds=$target -advval=1
