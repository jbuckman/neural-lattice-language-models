python get_data.py --language english --download

mv en/raw/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled en/raw/train
mv en/raw/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled en/raw/heldout

mkdir -p en/bpe
mkdir -p en/bpe/train
mkdir -p en/bpe/heldout

for i in $(seq -f "%02g" 1 99)
do
  python apply_bpe.py -c en/codes.txt --vocabulary en/bpe_vocab.txt --vocabulary-threshold 50 < en/raw/train/news.en-000$i-of-00100 > en/bpe/train/news.en-000$i-of-00100 &
done

for i in $(seq -f "%02g" 1 49)
do
  python apply_bpe.py -c en/codes.txt --vocabulary en/bpe_vocab.txt --vocabulary-threshold 50 < en/raw/heldout/news.en.heldout-000$i-of-00050 > en/bpe/heldout/news.en.heldout-000$i-of-00050 &
done

mkdir -p en/bpe/valid
mkdir -p en/bpe/test

chmod 777 . -R
cp en/bpe/heldout/* en/bpe/test
mv en/bpe/test/news.en.heldout-00001-of-00050 en/bpe/valid
