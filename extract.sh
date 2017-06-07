#! /bin/sh

for f in *.wav
do
	./SMILExtract -C MFCC12_E_D_A.conf -I "$f" -outputcsv "${f%.*}".csv
done
