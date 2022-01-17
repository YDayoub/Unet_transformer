echo "- Downloading WikiText-103 (WT103)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P ./datasets/
unzip -q datasets/wikitext-103-v1.zip -d datasets
rm datasets/wikitext-103-v1.zip
cd datasets/wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..
