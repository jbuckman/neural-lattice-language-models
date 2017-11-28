mkdir -p zh/raw
scp jacobbuc@tir.lti.cs.cmu.edu:chinese_data.tar zh/raw/raw_download.tar.gz
python get_data.py --language chinese

mv zh/raw/projects/lattice-lm/data/gmw/train.vocab.char zh/
mv zh/raw/projects/lattice-lm/data/gmw/train zh/
mv zh/raw/projects/lattice-lm/data/gmw/valid zh/
mv zh/raw/projects/lattice-lm/data/gmw/test zh/
mv zh/raw/projects/lattice-lm/data/gmw/vocabularies zh/