# P300 Speller with patients with ALS

This project is based on the concept of P300 ERP potential that arises due to an oddball stimulus. It could be used as a vital communication tool for people suffering with ALS or paralytic complications.

The 8 channel EEG data was digitized at 256 Hz and band-pass filtered between 0.1 and 10 Hz. Further it was segmented into 1 second long epochs.\
We tried various machine learning classifiers and Deep-CNN worked for us best, to classify the data into target or non target. Using this, we found the mode row and mode column, intersection of which gave us our letter.\
We were able to achieve nearly 95% accuracy in correctly predicting the character.

# Dataset & Reference

Dataset Link : http://bnci-horizon-2020.eu/database/data-sets \
Project/Dataset description : https://lampx.tugraz.at/~bci/database/008-2014/description.pdf \
Reference Paper : https://www.frontiersin.org/articles/10.3389/fnhum.2013.00732/full

# Data Structure
- 150 targets and 1050 non-targets. 
- bandpass  filtering from  0.5  Hz  to  30  Hz  and  decimation  by  a  factor  of 12.
- Trial  windows  were  set  to  0.1  s  pre-  and  0.7  s post-trigger,  yielding  15  samples  per  trial  and  channel. 
- The doublebarred flashes come 75 units before the P300 signal...


# Models to try
- Time-variant  linear  discriminant  analysis  (TVLDA)  
- Standard linear discriminant analysis (LDA)
  - 120-dimensional