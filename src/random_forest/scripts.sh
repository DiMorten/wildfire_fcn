
source='para'; target='acre';

source='acre'; target='para';


python random_forest_eachimage.py -ds=$source -mm='all_train'
python random_forest_eachimage.py -ds=$target -mm='all_test'



python random_forest_eachimage.py -ds=$source -mm='all_train' -tm='source_tval'
python random_forest_eachimage.py -ds=$target -mm='all_test' -tm='source_tval'


python random_forest_eachimage.py -ds=$source -mm='all_train' -tm='tval'
python random_forest_eachimage.py -ds=$target -mm='all_test' -tm='tval'

