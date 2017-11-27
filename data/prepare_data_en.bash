python get_data.py --language english --download

mkdir en/bpe
mkdir en/bpe/train
mkdir en/bpe/heldout

for i in $(seq -f "%02g" 1 99)
do
  python apply_bpe.py -c en/codes.txt --vocabulary en/bpe_vocab.txt --vocabulary-threshold 50 < en/raw/train/news.en-000$i-of-00100 > en/bpe/train/news.en-000$i-of-00100 &
done

for i in $(seq -f "%02g" 1 49)
do
  python apply_bpe.py -c en/codes.txt --vocabulary en/bpe_vocab.txt --vocabulary-threshold 50 < en/raw/heldout/news.en.heldout-000$i-of-00050 > en/bpe/heldout/news.en.heldout-000$i-of-00050 &
done

mkdir en/bpe/valid
mkdir en/bpe/test

cp en/bpe/heldout/* en/bpe/test
mv en/bpe/test/news.en.heldout-00001-of-00050 en/bpe/valid
