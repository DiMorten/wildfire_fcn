
source='para';
target='acre';

source='acre';
target='para';


python random_forest_eachimage.py -ds=$source -mm='all_train'
python random_forest_eachimage.py -ds=$target -mm='all_test'
