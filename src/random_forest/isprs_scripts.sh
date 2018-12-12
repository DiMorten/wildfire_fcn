
source="area23"; target="area3";

source='area3'; target='area23';


python random_forest_eachimage.py -ds=$source -mm='all_train' -bs=300000
python random_forest_eachimage.py -ds=$target -mm='all_test'



python random_forest_eachimage.py -ds=$source -mm='all_train' -tm='source_tval' -bs=300000
python random_forest_eachimage.py -ds=$target -mm='all_test' -tm='source_tval'


python random_forest_eachimage.py -ds=$source -mm='all_train' -tm='tval'
python random_forest_eachimage.py -ds=$target -mm='all_test' -tm='tval'

