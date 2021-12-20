echo "- Downloading WikiText-2 (WT2)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -P ./datasets/
unzip -q datasets/wikitext-2-v1.zip -d datasets
rm datasets/wikitext-2-v1.zip
cd datasets/wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..
