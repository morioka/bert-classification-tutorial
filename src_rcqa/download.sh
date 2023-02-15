mkdir -p data_rcqa

wget http://www.cl.ecei.tohoku.ac.jp/rcqa/data/all-v1.0.json.gz
mv ./all-v1.0.json.gz ./data_rcqa/
cd data_rcqa
gzip -dc ./all-v1.0.json.gz > all-v1.0.json
cd ..
